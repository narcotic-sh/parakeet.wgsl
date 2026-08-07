/*
 * Fixed-arena Parakeet log-mel frontend for standalone WebAssembly.
 *
 * The implementation follows the frozen model frontend contract:
 *   - 16 kHz mono float PCM
 *   - preemphasis 0.97 (first sample unchanged)
 *   - centered STFT with constant zero padding
 *   - 512-point FFT and a centered 400-point periodic-false Hann window
 *   - 128-bin Slaney-scale, Slaney-normalized mel filterbank
 *   - log(mel + 2^-24)
 *   - per-window, per-feature unbiased normalization; std += 1e-5
 *
 * No allocator is used. One worker owns one instance and reuses one fixed
 * input window, one fixed output feature item, and the raw-feature/FFT arenas.
 * JavaScript uploads each completed feature item before reusing the output.
 */

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include <emscripten/emscripten.h>
#include <wasm_simd128.h>

namespace {

constexpr int kSampleRate = 16'000;
constexpr int kFftSize = 512;
constexpr int kFftBins = kFftSize / 2 + 1;
constexpr int kWindowLength = 400;
constexpr int kWindowLeft = (kFftSize - kWindowLength) / 2;
constexpr int kHopLength = 160;
constexpr int kMelBins = 128;
constexpr int kMaxSamples = 240'000;
// A centered STFT has floor(samples / hop) + 1 physical frames. The model uses
// floor(valid_samples / hop) as the valid length and masks the final frame.
constexpr int kMaxFrames = 1'501;
static_assert(kMaxSamples / kHopLength + 1 == kMaxFrames);
constexpr int kMaxBatch = 40;
constexpr int kRmsFrameSamples = kHopLength * 8;
constexpr int kMaxRmsFrames = kMaxSamples / kRmsFrameSamples;
constexpr int kReuseBoundaryFrames = 3;
constexpr float kPreemphasis = 0.97f;
constexpr double kLogGuard = 1.0 / 16'777'216.0;  // 2^-24
constexpr double kNormalizeEpsilon = 1.0e-5;
constexpr double kPi = 3.14159265358979323846264338327950288;

// One item is decoded, transformed, and uploaded at a time. The logical batch
// can therefore grow without multiplying Wasm linear memory.
alignas(16) float input_samples[kMaxSamples];
alignas(16) float output_features[kMelBins * kMaxFrames];
// One complete input window contains at most 187 disjoint 80 ms RMS frames.
// JavaScript consumes these exact f32 bit patterns synchronously and retains
// the one bounded high-bit histogram across windows.
alignas(16) uint32_t rms_bits[kMaxRmsFrames];

// The raw cache belongs to the immediately preceding window, including across
// calls. Feature-major rows retain a fixed kMaxFrames stride so shifting a
// reusable interior never changes its addressing.
alignas(16) float raw_features[kMelBins * kMaxFrames];

alignas(16) float fft_real[kFftSize];
alignas(16) float fft_imaginary[kFftSize];
constexpr int kPackedFftFrames = 4;
alignas(16) float packed_fft_real[kFftSize][kPackedFftFrames];
alignas(16) float packed_fft_imaginary[kFftSize][kPackedFftFrames];
alignas(16) double power_spectrum[kFftBins];
alignas(16) float padded_hann[kFftSize];
alignas(16) float twiddle_real[kFftSize / 2];
alignas(16) float twiddle_imaginary[kFftSize / 2];
alignas(16) float mel_weights[kMelBins * kFftBins];
alignas(16) uint16_t bit_reverse[kFftSize];
alignas(16) uint16_t mel_first[kMelBins];
alignas(16) uint16_t mel_last_exclusive[kMelBins];

bool initialized = false;
bool previous_raw_valid = false;
int previous_sample_count = 0;
int previous_frame_count = 0;
int last_reused_frames = 0;
int active_batch_count = 0;
int active_sample_count = 0;
int active_physical_frame_count = 0;
int next_batch_index = 0;

inline double hz_to_slaney_mel(double frequency) {
  constexpr double kLinearSpacing = 200.0 / 3.0;
  constexpr double kLogTransitionHz = 1000.0;
  constexpr double kLogTransitionMel = kLogTransitionHz / kLinearSpacing;
  constexpr double kLogStep = 0.06875177742094912;  // log(6.4) / 27
  return frequency >= kLogTransitionHz
             ? kLogTransitionMel +
                   std::log(frequency / kLogTransitionHz) / kLogStep
             : frequency / kLinearSpacing;
}

inline double slaney_mel_to_hz(double mel) {
  constexpr double kLinearSpacing = 200.0 / 3.0;
  constexpr double kLogTransitionHz = 1000.0;
  constexpr double kLogTransitionMel = kLogTransitionHz / kLinearSpacing;
  constexpr double kLogStep = 0.06875177742094912;  // log(6.4) / 27
  return mel >= kLogTransitionMel
             ? kLogTransitionHz *
                   std::exp(kLogStep * (mel - kLogTransitionMel))
             : mel * kLinearSpacing;
}

void initialize_hann() {
  std::memset(padded_hann, 0, sizeof(padded_hann));
  for (int index = 0; index < kWindowLength; ++index) {
    const double phase =
        (2.0 * kPi * static_cast<double>(index)) /
        static_cast<double>(kWindowLength - 1);
    padded_hann[kWindowLeft + index] =
        static_cast<float>(0.5 * (1.0 - std::cos(phase)));
  }
}

void initialize_fft_tables() {
  for (int index = 0; index < kFftSize / 2; ++index) {
    const double phase =
        (-2.0 * kPi * static_cast<double>(index)) /
        static_cast<double>(kFftSize);
    twiddle_real[index] = static_cast<float>(std::cos(phase));
    twiddle_imaginary[index] = static_cast<float>(std::sin(phase));
  }

  for (int index = 0; index < kFftSize; ++index) {
    int source = index;
    int reversed = 0;
    for (int bit = 0; bit < 9; ++bit) {
      reversed = (reversed << 1) | (source & 1);
      source >>= 1;
    }
    bit_reverse[index] = static_cast<uint16_t>(reversed);
  }
}

void initialize_mel_filterbank() {
  constexpr int kPointCount = kMelBins + 2;
  double frequencies[kFftBins];
  double mel_points_hz[kPointCount];
  double point_differences[kPointCount - 1];

  for (int bin = 0; bin < kFftBins; ++bin) {
    frequencies[bin] =
        (static_cast<double>(kSampleRate) * 0.5 * bin) /
        static_cast<double>(kFftBins - 1);
  }

  const double mel_min = hz_to_slaney_mel(0.0);
  const double mel_max = hz_to_slaney_mel(kSampleRate * 0.5);
  for (int point = 0; point < kPointCount; ++point) {
    const double mel =
        mel_min + (mel_max - mel_min) * point / (kPointCount - 1);
    mel_points_hz[point] = slaney_mel_to_hz(mel);
  }
  for (int point = 0; point < kPointCount - 1; ++point) {
    point_differences[point] =
        mel_points_hz[point + 1] - mel_points_hz[point];
  }

  std::memset(mel_weights, 0, sizeof(mel_weights));
  for (int mel = 0; mel < kMelBins; ++mel) {
    const double normalization =
        2.0 / (mel_points_hz[mel + 2] - mel_points_hz[mel]);
    int first = -1;
    int last = -1;
    for (int bin = 0; bin < kFftBins; ++bin) {
      const double down =
          (frequencies[bin] - mel_points_hz[mel]) /
          point_differences[mel];
      const double up =
          (mel_points_hz[mel + 2] - frequencies[bin]) /
          point_differences[mel + 1];
      double triangle = down < up ? down : up;
      if (triangle < 0.0) triangle = 0.0;
      const float weight = static_cast<float>(triangle * normalization);
      mel_weights[mel * kFftBins + bin] = weight;
      if (weight > 0.0f) {
        if (first < 0) first = bin;
        last = bin;
      }
    }
    mel_first[mel] = static_cast<uint16_t>(first < 0 ? 0 : first);
    mel_last_exclusive[mel] =
        static_cast<uint16_t>(last < 0 ? 0 : last + 1);
  }
}

void initialize_tables() {
  initialize_hann();
  initialize_fft_tables();
  initialize_mel_filterbank();
}

void fft_in_place() {
  for (int length = 2; length <= kFftSize; length <<= 1) {
    const int half = length >> 1;
    const int twiddle_step = kFftSize / length;
    for (int block = 0; block < kFftSize; block += length) {
      // Every radix-2 block starts with W^0 = 1 - 0i. Handle that
      // butterfly directly, then preserve the arithmetic order for every
      // non-unity twiddle.
      const int unity_left = block;
      const int unity_right = block + half;
      const float unity_right_real = fft_real[unity_right];
      const float unity_right_imaginary = fft_imaginary[unity_right];
      const float unity_left_real = fft_real[unity_left];
      const float unity_left_imaginary = fft_imaginary[unity_left];
      fft_real[unity_left] = unity_left_real + unity_right_real;
      fft_imaginary[unity_left] =
          unity_left_imaginary + unity_right_imaginary;
      fft_real[unity_right] = unity_left_real - unity_right_real;
      fft_imaginary[unity_right] =
          unity_left_imaginary - unity_right_imaginary;

      int twiddle = twiddle_step;
      for (int lane = 1; lane < half; ++lane, twiddle += twiddle_step) {
        const int left = block + lane;
        const int right = left + half;
        const float right_real = fft_real[right];
        const float right_imaginary = fft_imaginary[right];
        const float rotated_real =
            twiddle_real[twiddle] * right_real -
            twiddle_imaginary[twiddle] * right_imaginary;
        const float rotated_imaginary =
            twiddle_real[twiddle] * right_imaginary +
            twiddle_imaginary[twiddle] * right_real;
        const float left_real = fft_real[left];
        const float left_imaginary = fft_imaginary[left];
        fft_real[left] = left_real + rotated_real;
        fft_imaginary[left] = left_imaginary + rotated_imaginary;
        fft_real[right] = left_real - rotated_real;
        fft_imaginary[right] = left_imaginary - rotated_imaginary;
      }
    }
  }
}

void prepare_frame(
    const float* samples,
    int sample_count,
    int frame) {
  std::memset(fft_real, 0, sizeof(fft_real));
  std::memset(fft_imaginary, 0, sizeof(fft_imaginary));

  const int center = frame * kHopLength;
  int source_start = center - kWindowLength / 2;
  if (source_start < 0) source_start = 0;
  int source_end = center + kWindowLength / 2;
  if (source_end > sample_count) source_end = sample_count;

  for (int source = source_start; source < source_end; ++source) {
    const double emphasized =
        source == 0
            ? static_cast<double>(samples[0])
            : static_cast<double>(samples[source]) -
                  static_cast<double>(kPreemphasis) *
                      static_cast<double>(samples[source - 1]);
    const int fft_index = source - center + kFftSize / 2;
    // The iterative FFT consumes bit-reversed input. Writing each prepared
    // value to that destination avoids a separate full-arena permutation.
    fft_real[bit_reverse[fft_index]] = static_cast<float>(
        emphasized * static_cast<double>(padded_hann[fft_index]));
  }
}

void compute_raw_frame(
    const float* samples,
    int sample_count,
    int frame) {
  prepare_frame(samples, sample_count, frame);
  fft_in_place();

  for (int bin = 0; bin < kFftBins; ++bin) {
    const double real = static_cast<double>(fft_real[bin]);
    const double imaginary = static_cast<double>(fft_imaginary[bin]);
    power_spectrum[bin] = real * real + imaginary * imaginary;
  }

  for (int mel = 0; mel < kMelBins; ++mel) {
    const float* weights = mel_weights + mel * kFftBins;
    double energy = 0.0;
    const int first = mel_first[mel];
    const int last = mel_last_exclusive[mel];
    for (int bin = first; bin < last; ++bin) {
      energy +=
          static_cast<double>(weights[bin]) * power_spectrum[bin];
    }
    raw_features[mel * kMaxFrames + frame] =
        static_cast<float>(std::log(energy + kLogGuard));
  }
}

void prepare_four_frames(
    const float* samples,
    int sample_count,
    int first_frame) {
  std::memset(packed_fft_real, 0, sizeof(packed_fft_real));
  std::memset(packed_fft_imaginary, 0, sizeof(packed_fft_imaginary));

  for (int frame_lane = 0; frame_lane < kPackedFftFrames; ++frame_lane) {
    const int frame = first_frame + frame_lane;
    const int center = frame * kHopLength;
    int source_start = center - kWindowLength / 2;
    if (source_start < 0) source_start = 0;
    int source_end = center + kWindowLength / 2;
    if (source_end > sample_count) source_end = sample_count;

    for (int source = source_start; source < source_end; ++source) {
      const double emphasized =
          source == 0
              ? static_cast<double>(samples[0])
              : static_cast<double>(samples[source]) -
                    static_cast<double>(kPreemphasis) *
                        static_cast<double>(samples[source - 1]);
      const int fft_index = source - center + kFftSize / 2;
      packed_fft_real[bit_reverse[fft_index]][frame_lane] =
          static_cast<float>(
              emphasized * static_cast<double>(padded_hann[fft_index]));
    }
  }
}

void fft_four_in_place() {
  for (int length = 2; length <= kFftSize; length <<= 1) {
    const int half = length >> 1;
    const int twiddle_step = kFftSize / length;
    for (int block = 0; block < kFftSize; block += length) {
      const int unity_left = block;
      const int unity_right = block + half;
      const v128_t unity_right_real =
          wasm_v128_load(packed_fft_real[unity_right]);
      const v128_t unity_right_imaginary =
          wasm_v128_load(packed_fft_imaginary[unity_right]);
      const v128_t unity_left_real =
          wasm_v128_load(packed_fft_real[unity_left]);
      const v128_t unity_left_imaginary =
          wasm_v128_load(packed_fft_imaginary[unity_left]);
      wasm_v128_store(
          packed_fft_real[unity_left],
          wasm_f32x4_add(unity_left_real, unity_right_real));
      wasm_v128_store(
          packed_fft_imaginary[unity_left],
          wasm_f32x4_add(
              unity_left_imaginary,
              unity_right_imaginary));
      wasm_v128_store(
          packed_fft_real[unity_right],
          wasm_f32x4_sub(unity_left_real, unity_right_real));
      wasm_v128_store(
          packed_fft_imaginary[unity_right],
          wasm_f32x4_sub(
              unity_left_imaginary,
              unity_right_imaginary));

      int twiddle = twiddle_step;
      for (int lane = 1; lane < half; ++lane, twiddle += twiddle_step) {
        const int left = block + lane;
        const int right = left + half;
        const v128_t right_real = wasm_v128_load(packed_fft_real[right]);
        const v128_t right_imaginary =
            wasm_v128_load(packed_fft_imaginary[right]);
        const v128_t twiddle_real_vector =
            wasm_f32x4_splat(twiddle_real[twiddle]);
        const v128_t twiddle_imaginary_vector =
            wasm_f32x4_splat(twiddle_imaginary[twiddle]);
        const v128_t rotated_real = wasm_f32x4_sub(
            wasm_f32x4_mul(twiddle_real_vector, right_real),
            wasm_f32x4_mul(
                twiddle_imaginary_vector,
                right_imaginary));
        const v128_t rotated_imaginary = wasm_f32x4_add(
            wasm_f32x4_mul(
                twiddle_real_vector,
                right_imaginary),
            wasm_f32x4_mul(
                twiddle_imaginary_vector,
                right_real));
        const v128_t left_real = wasm_v128_load(packed_fft_real[left]);
        const v128_t left_imaginary =
            wasm_v128_load(packed_fft_imaginary[left]);
        wasm_v128_store(
            packed_fft_real[left],
            wasm_f32x4_add(left_real, rotated_real));
        wasm_v128_store(
            packed_fft_imaginary[left],
            wasm_f32x4_add(left_imaginary, rotated_imaginary));
        wasm_v128_store(
            packed_fft_real[right],
            wasm_f32x4_sub(left_real, rotated_real));
        wasm_v128_store(
            packed_fft_imaginary[right],
            wasm_f32x4_sub(left_imaginary, rotated_imaginary));
      }
    }
  }
}

void compute_raw_four_frames(
    const float* samples,
    int sample_count,
    int first_frame) {
  prepare_four_frames(samples, sample_count, first_frame);
  fft_four_in_place();

  for (int frame_lane = 0; frame_lane < kPackedFftFrames; ++frame_lane) {
    for (int bin = 0; bin < kFftBins; ++bin) {
      const double real =
          static_cast<double>(packed_fft_real[bin][frame_lane]);
      const double imaginary =
          static_cast<double>(packed_fft_imaginary[bin][frame_lane]);
      power_spectrum[bin] = real * real + imaginary * imaginary;
    }

    const int frame = first_frame + frame_lane;
    for (int mel = 0; mel < kMelBins; ++mel) {
      const float* weights = mel_weights + mel * kFftBins;
      double energy = 0.0;
      const int first = mel_first[mel];
      const int last = mel_last_exclusive[mel];
      for (int bin = first; bin < last; ++bin) {
        energy +=
            static_cast<double>(weights[bin]) * power_spectrum[bin];
      }
      raw_features[mel * kMaxFrames + frame] =
          static_cast<float>(std::log(energy + kLogGuard));
    }
  }
}

void compute_raw_range(
    const float* samples,
    int sample_count,
    int first_frame,
    int last_frame) {
  int frame = first_frame;
  for (
      ;
      frame + kPackedFftFrames <= last_frame;
      frame += kPackedFftFrames) {
    compute_raw_four_frames(samples, sample_count, frame);
  }
  for (; frame < last_frame; ++frame) {
    compute_raw_frame(samples, sample_count, frame);
  }
}

int reuse_previous_interior(
    int sample_count,
    int frame_count,
    int frame_shift,
    int& reused_start) {
  reused_start = 0;
  if (!previous_raw_valid ||
      previous_sample_count != sample_count ||
      frame_shift <= 0 ||
      frame_shift >= previous_frame_count) {
    return 0;
  }

  // Center padding and window-local preemphasis change the boundary frames
  // even when two windows share an aligned PCM overlap. Recompute a conservative
  // three-frame margin at both changed edges.
  const int old_start = frame_shift + kReuseBoundaryFrames;
  int old_end = previous_frame_count - kReuseBoundaryFrames;
  const int target_limited_end =
      frame_shift + frame_count - kReuseBoundaryFrames;
  if (old_end > target_limited_end) old_end = target_limited_end;
  if (old_end <= old_start) return 0;

  reused_start = old_start - frame_shift;
  const int reused_count = old_end - old_start;
  for (int mel = 0; mel < kMelBins; ++mel) {
    float* row = raw_features + mel * kMaxFrames;
    std::memmove(
        row + reused_start,
        row + old_start,
        static_cast<size_t>(reused_count) * sizeof(float));
  }
  return reused_count;
}

void normalize_window(int physical_frame_count, int valid_frame_count) {
  for (int mel = 0; mel < kMelBins; ++mel) {
    const float* raw = raw_features + mel * kMaxFrames;
    float* normalized = output_features + mel * physical_frame_count;

    double sum = 0.0;
#pragma clang loop vectorize(enable) interleave(enable)
    for (int frame = 0; frame < valid_frame_count; ++frame) {
      sum += static_cast<double>(raw[frame]);
    }
    const double mean =
        valid_frame_count > 0
            ? sum / static_cast<double>(valid_frame_count)
            : 0.0;

    double squared_deviation = 0.0;
#pragma clang loop vectorize(enable) interleave(enable)
    for (int frame = 0; frame < valid_frame_count; ++frame) {
      const double difference = static_cast<double>(raw[frame]) - mean;
      squared_deviation += difference * difference;
    }
    const double standard_deviation =
        valid_frame_count > 1
            ? std::sqrt(
                  squared_deviation /
                  static_cast<double>(valid_frame_count - 1))
            : 0.0;
    const double inverse =
        1.0 / (standard_deviation + kNormalizeEpsilon);

#pragma clang loop vectorize(enable) interleave(enable)
    for (int frame = 0; frame < valid_frame_count; ++frame) {
      normalized[frame] = static_cast<float>(
          (static_cast<double>(raw[frame]) - mean) * inverse);
    }
    std::memset(
        normalized + valid_frame_count,
        0,
        static_cast<size_t>(physical_frame_count - valid_frame_count) *
            sizeof(float));
  }
}

void clear_reuse_state() {
  previous_raw_valid = false;
  previous_sample_count = 0;
  previous_frame_count = 0;
  last_reused_frames = 0;
}

void clear_active_batch() {
  active_batch_count = 0;
  active_sample_count = 0;
  active_physical_frame_count = 0;
  next_batch_index = 0;
}

}  // namespace

extern "C" {

EMSCRIPTEN_KEEPALIVE int parakeet_fbank_init() {
  if (!initialized) {
    initialize_tables();
    initialized = true;
  }
  clear_reuse_state();
  clear_active_batch();
  return 1;
}

EMSCRIPTEN_KEEPALIVE uintptr_t parakeet_fbank_input_ptr() {
  return reinterpret_cast<uintptr_t>(input_samples);
}

EMSCRIPTEN_KEEPALIVE uintptr_t parakeet_fbank_output_ptr() {
  return reinterpret_cast<uintptr_t>(output_features);
}

EMSCRIPTEN_KEEPALIVE uintptr_t parakeet_fbank_rms_bits_ptr() {
  return reinterpret_cast<uintptr_t>(rms_bits);
}

EMSCRIPTEN_KEEPALIVE int parakeet_fbank_compute_rms_bits(
    int sample_offset,
    int frame_count) {
  if (!initialized) return -1;
  if (active_batch_count == 0) return -6;
  if (sample_offset < 0 ||
      frame_count < 0 ||
      frame_count > kMaxRmsFrames ||
      sample_offset + frame_count * kRmsFrameSamples >
          active_sample_count) {
    return -9;
  }

  // This is the exact ordered Float32 reduction used by the FluidAudio-derived
  // TypeScript policy. Reassociation, vector reduction, or contraction would
  // be faster but could move an RMS value to another radix bucket.
  {
#pragma clang fp reassociate(off)
#pragma clang fp contract(off)
    for (int frame = 0; frame < frame_count; ++frame) {
      const float* samples =
          input_samples + sample_offset + frame * kRmsFrameSamples;
      float sum = 0.0f;
#pragma clang loop vectorize(disable) interleave(disable)
      for (int index = 0; index < kRmsFrameSamples; ++index) {
        const float squared = samples[index] * samples[index];
        sum = sum + squared;
      }
      const float rms =
          std::sqrt(sum / static_cast<float>(kRmsFrameSamples));
      rms_bits[frame] = std::bit_cast<uint32_t>(rms);
    }
  }
  return frame_count;
}

EMSCRIPTEN_KEEPALIVE int parakeet_fbank_begin_batch(
    int batch_count,
    int sample_count) {
  if (!initialized) return -1;
  if (batch_count < 1 || batch_count > kMaxBatch) return -2;
  if (sample_count < kHopLength || sample_count > kMaxSamples) {
    return -3;
  }
  if (active_batch_count != 0) return -6;

  const int physical_frame_count =
      sample_count / kHopLength + 1;
  if (physical_frame_count < 2 || physical_frame_count > kMaxFrames) return -4;

  active_batch_count = batch_count;
  active_sample_count = sample_count;
  active_physical_frame_count = physical_frame_count;
  next_batch_index = 0;
  last_reused_frames = 0;
  return physical_frame_count;
}

EMSCRIPTEN_KEEPALIVE int parakeet_fbank_compute_item(
    int batch_index,
    int valid_sample_count,
    int reuse_frame_shift) {
  if (!initialized) return -1;
  if (active_batch_count == 0) return -6;
  if (batch_index != next_batch_index ||
      batch_index < 0 ||
      batch_index >= active_batch_count) {
    return -7;
  }
  if (valid_sample_count < 0 ||
      valid_sample_count > active_sample_count) {
    return -5;
  }

  const int valid_frame_count =
      valid_sample_count / kHopLength;
  if (reuse_frame_shift < 0 ||
      (reuse_frame_shift != 0 &&
       reuse_frame_shift >= valid_frame_count)) {
    return -8;
  }

  int reused_start = 0;
  const int reused_count = reuse_previous_interior(
      active_sample_count,
      valid_frame_count,
      reuse_frame_shift,
      reused_start);
  const int reused_end = reused_start + reused_count;

  // The model masks preemphasis at the declared sequence length. Splitting at
  // the reused interior also keeps each contiguous SIMD group independent of
  // moved raw-feature rows.
  compute_raw_range(
      input_samples,
      valid_sample_count,
      0,
      reused_start);
  compute_raw_range(
      input_samples,
      valid_sample_count,
      reused_end,
      valid_frame_count);
  normalize_window(active_physical_frame_count, valid_frame_count);
  last_reused_frames += reused_count;

  previous_raw_valid = true;
  previous_sample_count = active_sample_count;
  previous_frame_count = valid_frame_count;
  next_batch_index += 1;
  return active_physical_frame_count;
}

EMSCRIPTEN_KEEPALIVE int parakeet_fbank_finish_batch() {
  if (!initialized) return -1;
  if (active_batch_count == 0) return -6;
  if (next_batch_index != active_batch_count) return -7;
  const int physical_frame_count = active_physical_frame_count;
  clear_active_batch();
  return physical_frame_count;
}

EMSCRIPTEN_KEEPALIVE void parakeet_fbank_reset_reuse() {
  clear_reuse_state();
  clear_active_batch();
}

EMSCRIPTEN_KEEPALIVE void parakeet_fbank_dispose() {
  clear_reuse_state();
  clear_active_batch();
}

EMSCRIPTEN_KEEPALIVE int parakeet_fbank_last_reused_frames() {
  return last_reused_frames;
}

EMSCRIPTEN_KEEPALIVE int parakeet_fbank_max_batch() {
  return kMaxBatch;
}

EMSCRIPTEN_KEEPALIVE int parakeet_fbank_max_samples() {
  return kMaxSamples;
}

EMSCRIPTEN_KEEPALIVE int parakeet_fbank_max_frames() {
  return kMaxFrames;
}

EMSCRIPTEN_KEEPALIVE int parakeet_fbank_bins() {
  return kMelBins;
}

EMSCRIPTEN_KEEPALIVE int parakeet_fbank_sample_rate() {
  return kSampleRate;
}

EMSCRIPTEN_KEEPALIVE int parakeet_fbank_hop_length() {
  return kHopLength;
}

EMSCRIPTEN_KEEPALIVE int parakeet_fbank_rms_frame_samples() {
  return kRmsFrameSamples;
}

EMSCRIPTEN_KEEPALIVE int parakeet_fbank_max_rms_frames() {
  return kMaxRmsFrames;
}

}  // extern "C"
