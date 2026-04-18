import type { ReactNode } from "react";

interface Props {
  text: string;
  children: ReactNode;
  showIcon?: boolean;
}

export function Tip({ text, children, showIcon = true }: Props) {
  return (
    <span className="tip-target" data-tip={text} tabIndex={0}>
      {children}
      {showIcon && <span className="tip-icon">?</span>}
    </span>
  );
}
