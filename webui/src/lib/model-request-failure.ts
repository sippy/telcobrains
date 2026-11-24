import type { TFunction } from "i18next";

export interface ModelRequestFailureDetails {
  errorKind?: string;
  attempts?: number;
  /** Clean limit-reset clock time for a usage-limit failure (e.g. "1pm"). */
  resetTime?: string;
  /** IANA time zone for a usage-limit failure (e.g. "America/Vancouver"). */
  resetTimezone?: string;
}

export function isImageRequestFailure(errorKind?: string): boolean {
  return errorKind === "unsupported_image" || errorKind === "image_request_rejected";
}

export function resolveModelRequestFailureCopy(
  details: ModelRequestFailureDetails,
  t: TFunction,
): { title: string; body: string } {
  if (isImageRequestFailure(details.errorKind)) {
    return {
      title: t(details.errorKind === "unsupported_image"
        ? "errors.modelRequestFailed.imageInputTitle"
        : "errors.modelRequestFailed.imageRequestRejectedTitle"),
      body: t("errors.modelRequestFailed.imageInputBody"),
    };
  }
  let titleKey = "errors.modelRequestFailed.unknownTitle";
  if (details.errorKind === "billing") {
    titleKey = "errors.modelRequestFailed.billingTitle";
  } else if (details.errorKind === "authentication") {
    titleKey = "errors.modelRequestFailed.authenticationTitle";
  } else if (details.errorKind === "connection") {
    titleKey = "errors.modelRequestFailed.connectionTitle";
  } else if (details.errorKind === "timeout") {
    titleKey = "errors.modelRequestFailed.timeoutTitle";
  } else if (details.errorKind === "rate_limit") {
    titleKey = "errors.modelRequestFailed.rateLimitTitle";
  } else if (details.errorKind === "usage_limit") {
    titleKey = "errors.modelRequestFailed.usageLimitTitle";
  } else if (details.errorKind === "server") {
    titleKey = "errors.modelRequestFailed.serverTitle";
  }
  const attempts = details.attempts;
  let bodyKey = "errors.modelRequestFailed.body";
  if (details.errorKind === "usage_limit") {
    bodyKey = details.resetTime && details.resetTimezone
      ? "errors.modelRequestFailed.usageLimitBodyWithReset"
      : "errors.modelRequestFailed.usageLimitBody";
  } else if (details.errorKind === "billing") {
    bodyKey = "errors.modelRequestFailed.billingBody";
  } else if (details.errorKind === "authentication") {
    bodyKey = "errors.modelRequestFailed.authenticationBody";
  } else if (typeof attempts === "number" && Number.isInteger(attempts) && attempts > 0) {
    bodyKey = "errors.modelRequestFailed.bodyWithAttempt";
  }

  return {
    title: t(titleKey),
    body: t(bodyKey, { attempts, resetTime: details.resetTime, timeZone: details.resetTimezone }),
  };
}
