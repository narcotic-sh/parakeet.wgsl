import { describe, expect, it } from "vitest";

import {
  FIREFOX_PERFORMANCE_WARNING,
  OTHER_BROWSER_PERFORMANCE_WARNING,
  performanceWarningForBrowser,
} from "../demo/src/browser-performance-warning";

const CHROME_USER_AGENT =
  "Mozilla/5.0 AppleWebKit/537.36 Chrome/151.0.0.0 Safari/537.36";

describe("demo browser performance warning", () => {
  it("uses the Firefox-specific warning on desktop and iOS", () => {
    expect(
      performanceWarningForBrowser({
        userAgent: "Mozilla/5.0 Firefox/147.0",
      }),
    ).toBe(FIREFOX_PERFORMANCE_WARNING);
    expect(
      performanceWarningForBrowser({
        userAgent: "Mozilla/5.0 FxiOS/147.0 Mobile/15E148 Safari/605.1.15",
        vendor: "Apple Computer, Inc.",
      }),
    ).toBe(FIREFOX_PERFORMANCE_WARNING);
  });

  it("stays hidden in Chrome on desktop and iOS", () => {
    expect(
      performanceWarningForBrowser({
        userAgent: CHROME_USER_AGENT,
        vendor: "Google Inc.",
        userAgentData: {
          brands: [
            { brand: "Chromium" },
            { brand: "Google Chrome" },
          ],
        },
      }),
    ).toBeUndefined();
    expect(
      performanceWarningForBrowser({
        userAgent: "Mozilla/5.0 CriOS/151.0 Mobile/15E148 Safari/604.1",
        vendor: "Apple Computer, Inc.",
      }),
    ).toBeUndefined();
  });

  it("stays hidden in Safari on macOS and iOS", () => {
    expect(
      performanceWarningForBrowser({
        userAgent:
          "Mozilla/5.0 Version/26.0 Safari/620.1.15 AppleWebKit/620.1.15",
        vendor: "Apple Computer, Inc.",
      }),
    ).toBeUndefined();
    expect(
      performanceWarningForBrowser({
        userAgent:
          "Mozilla/5.0 iPhone Version/26.0 Mobile/15E148 Safari/604.1",
        vendor: "Apple Computer, Inc.",
      }),
    ).toBeUndefined();
  });

  it("uses the generic warning for identifiable Chrome derivatives", () => {
    for (const userAgent of [
      `${CHROME_USER_AGENT} Edg/151.0`,
      `${CHROME_USER_AGENT} OPR/122.0`,
      `${CHROME_USER_AGENT} SamsungBrowser/29.0`,
      `${CHROME_USER_AGENT} Vivaldi/7.0`,
      "Mozilla/5.0; wv) AppleWebKit/537.36 Chrome/151.0 Mobile Safari/537.36",
    ]) {
      expect(performanceWarningForBrowser({ userAgent })).toBe(
        OTHER_BROWSER_PERFORMANCE_WARNING,
      );
    }
    expect(
      performanceWarningForBrowser({
        userAgent: CHROME_USER_AGENT,
        userAgentData: { brands: [{ brand: "Chromium" }] },
      }),
    ).toBe(OTHER_BROWSER_PERFORMANCE_WARNING);
    expect(
      performanceWarningForBrowser({
        userAgent: CHROME_USER_AGENT,
        brave: {},
      }),
    ).toBe(OTHER_BROWSER_PERFORMANCE_WARNING);
  });

  it("uses the generic warning for unknown browsers", () => {
    expect(
      performanceWarningForBrowser({ userAgent: "ExampleBrowser/1.0" }),
    ).toBe(OTHER_BROWSER_PERFORMANCE_WARNING);
  });
});
