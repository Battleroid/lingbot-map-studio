---
name: vid3d-studio-design
description: Use this skill to generate well-branded interfaces and assets for vid3d Studio (lingbot-map-studio), a monospace, grayscale-themed video-to-3D/gaussian-splat application. Contains essential design guidelines, colors, type, fonts, assets, and UI kit components for prototyping.
user-invocable: true
---

Read the README.md file within this skill, and explore the other available files.

If creating visual artifacts (slides, mocks, throwaway prototypes, etc), copy assets out and create static HTML files for the user to view. If working on production code, you can copy assets and read the rules here to become an expert in designing with this brand.

If the user invokes this skill without any other guidance, ask them what they want to build or design, ask some questions, and act as an expert designer who outputs HTML artifacts _or_ production code, depending on the need.

Key invariants:
- Mono only. Commit Mono bundled; Berkeley Mono licensed drop-in.
- Four grays + one danger. No other hues in chrome.
- 1px black rules, 0 radius, no shadows (except the `0 2px 0 fg` tooltip offset).
- Lowercase UI chrome; UPPERCASE only for titles/headers/acronyms.
- Dithered 2×2 checker backgrounds, never gradients.
- Zero icons. Use unicode `· — ✓ ✕` and short word labels.
- Narrow type ladder: 11 / 12 / 13 / 16 / 22.
- Animation is sparse and functional: stripe, blink cursor, stage pulse, VRAM flash.
