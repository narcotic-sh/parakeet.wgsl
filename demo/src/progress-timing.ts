export function formatProgressElapsedMilliseconds(
  milliseconds: number,
): string {
  const safeMilliseconds =
    Number.isFinite(milliseconds) && milliseconds > 0 ? milliseconds : 0;
  const totalSeconds = Math.floor(safeMilliseconds / 1_000);
  const seconds = totalSeconds % 60;
  const totalMinutes = Math.floor(totalSeconds / 60);
  const minutes = totalMinutes % 60;
  const hours = Math.floor(totalMinutes / 60);
  const clock =
    hours > 0
      ? `${hours}:${pad2(minutes)}:${pad2(seconds)}`
      : `${totalMinutes}:${pad2(seconds)}`;
  return `${clock} elapsed`;
}

function pad2(value: number): string {
  return value.toString().padStart(2, "0");
}
