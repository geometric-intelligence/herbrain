/// <reference path="../.astro/types.d.ts" />
/// <reference types="astro/client" />

interface ImportMetaEnv {
  readonly PUBLIC_R2_BUCKET_URL: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
