import type { SVGProps } from "react";

type P = SVGProps<SVGSVGElement> & { size?: number };

const base = (size = 20): SVGProps<SVGSVGElement> => ({
  width: size,
  height: size,
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.8,
  strokeLinecap: "round",
  strokeLinejoin: "round",
  "aria-hidden": true,
});

export const IconGauge = ({ size, ...p }: P) => (
  <svg {...base(size)} {...p}>
    <path d="M4 17a8 8 0 1 1 16 0" />
    <path d="M12 17l4.5-5" />
    <path d="M3 21h18" />
  </svg>
);

export const IconCamera = ({ size, ...p }: P) => (
  <svg {...base(size)} {...p}>
    <path d="M4 8h3l2-3h6l2 3h3v11H4z" />
    <circle cx="12" cy="13" r="3.5" />
  </svg>
);

export const IconPallet = ({ size, ...p }: P) => (
  <svg {...base(size)} {...p}>
    <rect x="4" y="4" width="7" height="7" />
    <rect x="13" y="4" width="7" height="7" />
    <rect x="4" y="11" width="16" height="4" />
    <path d="M4 15v4M12 15v4M20 15v4M3 19h18" />
  </svg>
);

export const IconReview = ({ size, ...p }: P) => (
  <svg {...base(size)} {...p}>
    <path d="M9 4H5v16h14v-4" />
    <path d="M9 12l2.5 2.5L20 6" />
  </svg>
);

export const IconSliders = ({ size, ...p }: P) => (
  <svg {...base(size)} {...p}>
    <path d="M5 4v16M12 4v16M19 4v16" />
    <rect x="3" y="13" width="4" height="3" fill="currentColor" />
    <rect x="10" y="7" width="4" height="3" fill="currentColor" />
    <rect x="17" y="15" width="4" height="3" fill="currentColor" />
  </svg>
);

export const IconCheck = ({ size = 14, ...p }: P) => (
  <svg {...base(size)} strokeWidth={2.6} {...p}>
    <path d="M5 12.5l4.5 4.5L19 7" />
  </svg>
);

export const IconCross = ({ size = 14, ...p }: P) => (
  <svg {...base(size)} strokeWidth={2.6} {...p}>
    <path d="M6 6l12 12M18 6L6 18" />
  </svg>
);

export const IconEye = ({ size = 14, ...p }: P) => (
  <svg {...base(size)} strokeWidth={2.4} {...p}>
    <path d="M2 12s3.5-6 10-6 10 6 10 6-3.5 6-10 6S2 12 2 12z" />
    <circle cx="12" cy="12" r="2.5" />
  </svg>
);

export const IconUpload = ({ size, ...p }: P) => (
  <svg {...base(size)} {...p}>
    <path d="M12 16V4M7 9l5-5 5 5" />
    <path d="M4 16v4h16v-4" />
  </svg>
);

export const IconDownload = ({ size = 16, ...p }: P) => (
  <svg {...base(size)} {...p}>
    <path d="M12 4v12M7 11l5 5 5-5" />
    <path d="M4 20h16" />
  </svg>
);

export const IconPrint = ({ size = 16, ...p }: P) => (
  <svg {...base(size)} {...p}>
    <path d="M7 9V4h10v5" />
    <rect x="4" y="9" width="16" height="7" />
    <path d="M7 14h10v6H7z" />
  </svg>
);

export const IconChevron = ({ size = 16, ...p }: P) => (
  <svg {...base(size)} {...p}>
    <path d="M9 6l6 6-6 6" />
  </svg>
);
