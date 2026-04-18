"use client";

import { useState, type ReactNode } from "react";

interface Props {
  title: string;
  /** Optional right-side metadata rendered in the header (e.g. "5 files"). */
  meta?: ReactNode;
  /** When true, clicking the header toggles collapsed state. Default true. */
  collapsible?: boolean;
  /** Initial collapsed state. Default false (expanded). */
  defaultCollapsed?: boolean;
  children: ReactNode;
  /** Extra style for the body (e.g. padding: 0). */
  bodyStyle?: React.CSSProperties;
  /** Extra classes for the body. */
  bodyClassName?: string;
}

/**
 * Standard panel shell. Clicking the header toggles the body's visibility —
 * panels you aren't actively using can be collapsed so the remaining ones
 * get their full natural height in the side column.
 */
export function Panel({
  title,
  meta,
  collapsible = true,
  defaultCollapsed = false,
  children,
  bodyStyle,
  bodyClassName,
}: Props) {
  const [collapsed, setCollapsed] = useState(defaultCollapsed);
  return (
    <div className="panel" data-collapsed={collapsed}>
      <div
        className="panel-header"
        data-collapsible={collapsible}
        onClick={collapsible ? () => setCollapsed((v) => !v) : undefined}
        role={collapsible ? "button" : undefined}
        tabIndex={collapsible ? 0 : undefined}
        onKeyDown={
          collapsible
            ? (e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  setCollapsed((v) => !v);
                }
              }
            : undefined
        }
      >
        <span>
          {collapsible && <span className="collapse-arrow">▾</span>}
          {title}
        </span>
        {meta != null && <span className="meta">{meta}</span>}
      </div>
      <div className={`panel-body ${bodyClassName ?? ""}`} style={bodyStyle}>
        {children}
      </div>
    </div>
  );
}
