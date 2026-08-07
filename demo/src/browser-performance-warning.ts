export const FIREFOX_PERFORMANCE_WARNING =
  "Warning: runs fastest in Chrome and Safari, much slower in Firefox.";

export const OTHER_BROWSER_PERFORMANCE_WARNING =
  "Warning: runs fastest in Chrome and Safari, much slower in other browsers.";

interface UserAgentBrand {
  readonly brand: string;
}

export interface BrowserIdentity {
  readonly userAgent: string;
  readonly vendor?: string;
  readonly userAgentData?: {
    readonly brands?: readonly UserAgentBrand[];
  };
  readonly brave?: unknown;
}

const FIREFOX_PATTERN = /(?:Firefox|FxiOS)\//;
const CHROME_PATTERN = /(?:Chrome|CriOS)\//;
const SAFARI_PATTERN = /Safari\//;
const SAFARI_VERSION_PATTERN = /Version\//;
const KNOWN_OTHER_BROWSER_PATTERN =
  /(?:Edg|EdgA|EdgiOS|Edge|OPR|OPiOS|Vivaldi|SamsungBrowser|YaBrowser|DuckDuckGo|GSA|Electron)\//;
const ANDROID_WEBVIEW_PATTERN = /;\s*wv\)/;

export function performanceWarningForBrowser(
  identity: BrowserIdentity,
): string | undefined {
  const userAgent = identity.userAgent;
  if (FIREFOX_PATTERN.test(userAgent)) {
    return FIREFOX_PERFORMANCE_WARNING;
  }

  const brands = identity.userAgentData?.brands?.map(({ brand }) => brand) ?? [];
  const isKnownOtherBrowser =
    identity.brave !== undefined ||
    KNOWN_OTHER_BROWSER_PATTERN.test(userAgent) ||
    ANDROID_WEBVIEW_PATTERN.test(userAgent);
  if (isKnownOtherBrowser) {
    return OTHER_BROWSER_PERFORMANCE_WARNING;
  }

  const isChrome =
    brands.includes("Google Chrome") ||
    (brands.length === 0 && CHROME_PATTERN.test(userAgent));
  if (isChrome) return undefined;

  const isSafari =
    SAFARI_PATTERN.test(userAgent) &&
    (identity.vendor === "Apple Computer, Inc." ||
      SAFARI_VERSION_PATTERN.test(userAgent));
  return isSafari ? undefined : OTHER_BROWSER_PERFORMANCE_WARNING;
}
