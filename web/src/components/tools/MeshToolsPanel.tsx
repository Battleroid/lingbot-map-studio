"use client";

/**
 * Mesh-editing tools. Re-export of the existing MeshTools component under
 * the tools/ namespace. Kept as a thin wrapper so job pages always import
 * from `tools/*` and the dispatcher can swap in a different panel later
 * without touching every call site.
 */
export { MeshTools as MeshToolsPanel } from "@/components/MeshTools";
