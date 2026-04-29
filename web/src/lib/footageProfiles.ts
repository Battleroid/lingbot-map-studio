import type { PreprocFields } from "@/lib/types";

/**
 * Footage profiles — UI sugar that bundles the `preproc_*` toggles into
 * a single high-level pick. The profile is *derived* from the bools, not
 * stored — so we can show the right pick on a fresh draft, after the
 * server-side probe heuristic auto-enables the analog bundle, or after
 * the user touches an individual toggle.
 *
 * `hi_def` is the no-op profile and the explicit default. A user dropping
 * a phone clip sees "hi-def · no preprocessing" and knows nothing is being
 * mangled. Picking `fpv_analog` or `fpv_aggressive` flips the bool bundle
 * in one click; touching a single toggle from there flips the dropdown to
 * `custom` so it never lies about the underlying state.
 */
export type FootageProfile = "hi_def" | "fpv_analog" | "fpv_aggressive" | "custom";

export interface FootageProfileMeta {
  id: FootageProfile;
  label: string;
  desc: string;
}

/** Display order for the dropdown. `custom` is appended dynamically when
 *  the bools don't match a named profile. */
export const FOOTAGE_PROFILE_ORDER: FootageProfile[] = [
  "hi_def",
  "fpv_analog",
  "fpv_aggressive",
];

export const FOOTAGE_PROFILES: Record<FootageProfile, FootageProfileMeta> = {
  hi_def: {
    id: "hi_def",
    label: "hi-def · no preproc",
    desc: "Default. Footage from a phone, mirrorless, action cam, or HD drone — no cleanup applied. Drop your clip in raw and let the reconstruction model see it as-is.",
  },
  fpv_analog: {
    id: "fpv_analog",
    label: "fpv · analog",
    desc: "Bundle for low-bitrate analog FPV captures (DVR rips, analog receivers): temporal denoise, deflicker, OSD/HUD masking, white-balance, rolling-shutter correction, unsharp deblur, keyframe scoring.",
  },
  fpv_aggressive: {
    id: "fpv_aggressive",
    label: "fpv · aggressive",
    desc: "Same as fpv · analog, plus heavier atadenoise temporal cleanup. Slow; reserve for visibly rough captures where the standard bundle still leaves chroma noise or dot crawl.",
  },
  custom: {
    id: "custom",
    label: "custom",
    desc: "Manual mix — your toggle picks don't match any named profile. Use the advanced toggles below to fine-tune.",
  },
};

/** Subset of PreprocFields the profile system manages. Each profile maps
 *  every field to a concrete value so applying it is a clean overwrite. */
type ManagedKey =
  | "preproc_denoise"
  | "preproc_analog_cleanup"
  | "preproc_deflicker"
  | "preproc_osd_mask"
  | "preproc_color_norm"
  | "preproc_rs_correction"
  | "preproc_deblur"
  | "preproc_keyframe_score";

const MANAGED_KEYS: ManagedKey[] = [
  "preproc_denoise",
  "preproc_analog_cleanup",
  "preproc_deflicker",
  "preproc_osd_mask",
  "preproc_color_norm",
  "preproc_rs_correction",
  "preproc_deblur",
  "preproc_keyframe_score",
];

type ProfilePatch = Pick<PreprocFields, ManagedKey>;

const HI_DEF_PATCH: ProfilePatch = {
  preproc_denoise: false,
  preproc_analog_cleanup: false,
  preproc_deflicker: false,
  preproc_osd_mask: false,
  preproc_color_norm: false,
  preproc_rs_correction: false,
  preproc_deblur: "none",
  preproc_keyframe_score: false,
};

const FPV_ANALOG_PATCH: ProfilePatch = {
  preproc_denoise: true,
  preproc_analog_cleanup: false,
  preproc_deflicker: true,
  preproc_osd_mask: true,
  preproc_color_norm: true,
  preproc_rs_correction: true,
  preproc_deblur: "unsharp",
  preproc_keyframe_score: true,
};

const FPV_AGGRESSIVE_PATCH: ProfilePatch = {
  ...FPV_ANALOG_PATCH,
  preproc_analog_cleanup: true,
};

/** Profile id → field patch the dropdown writes to the parent config. */
export const FOOTAGE_PROFILE_PATCHES: Record<
  Exclude<FootageProfile, "custom">,
  ProfilePatch
> = {
  hi_def: HI_DEF_PATCH,
  fpv_analog: FPV_ANALOG_PATCH,
  fpv_aggressive: FPV_AGGRESSIVE_PATCH,
};

/**
 * Find the named profile whose managed-field values exactly match the
 * current config. Returns `"custom"` when no profile matches — used by
 * the dropdown so manual edits don't leave the picker showing a
 * misleading label.
 */
export function deriveFootageProfile(
  config: Pick<PreprocFields, ManagedKey>,
): FootageProfile {
  for (const id of FOOTAGE_PROFILE_ORDER) {
    if (id === "custom") continue;
    const patch = FOOTAGE_PROFILE_PATCHES[id];
    let matches = true;
    for (const key of MANAGED_KEYS) {
      if (config[key] !== patch[key]) {
        matches = false;
        break;
      }
    }
    if (matches) return id;
  }
  return "custom";
}
