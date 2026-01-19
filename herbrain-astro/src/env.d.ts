/// <reference path="../.astro/types.d.ts" />
/// <reference types="astro/client" />

interface ImportMetaEnv {
  readonly PUBLIC_R2_BUCKET_URL: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}

// Type declaration for plotly.js-dist-min
declare module 'plotly.js-dist-min' {
  import Plotly from 'plotly.js';
  export default Plotly;
}
