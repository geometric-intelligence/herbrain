/**
 * NIfTI parsing utilities for browser-based MRI visualization
 */
import * as nifti from 'nifti-reader-js';
import { getCachedVolume, cacheVolume } from './storage';

export interface NiftiVolume {
  data: Float32Array;
  dims: [number, number, number];
  affine: number[][];
}

/**
 * Load and parse a NIfTI file from a URL
 */
export async function loadNiftiFromUrl(url: string): Promise<NiftiVolume> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch NIfTI file: ${response.statusText}`);
  }
  
  let arrayBuffer = await response.arrayBuffer();
  
  // Decompress if gzipped
  if (nifti.isCompressed(arrayBuffer)) {
    arrayBuffer = nifti.decompress(arrayBuffer) as ArrayBuffer;
  }
  
  if (!nifti.isNIFTI(arrayBuffer)) {
    throw new Error('Not a valid NIfTI file');
  }
  
  const header = nifti.readHeader(arrayBuffer);
  if (!header) {
    throw new Error('Failed to read NIfTI header');
  }
  
  const imageData = nifti.readImage(header, arrayBuffer);
  
  // Convert to Float32Array for consistent handling
  const dims: [number, number, number] = [
    header.dims[1],
    header.dims[2],
    header.dims[3],
  ];
  
  // Handle different data types
  let typedData: Float32Array;
  switch (header.datatypeCode) {
    case nifti.NIFTI1.TYPE_UINT8:
      typedData = new Float32Array(new Uint8Array(imageData));
      break;
    case nifti.NIFTI1.TYPE_INT16:
      typedData = new Float32Array(new Int16Array(imageData));
      break;
    case nifti.NIFTI1.TYPE_INT32:
      typedData = new Float32Array(new Int32Array(imageData));
      break;
    case nifti.NIFTI1.TYPE_FLOAT32:
      typedData = new Float32Array(imageData);
      break;
    case nifti.NIFTI1.TYPE_FLOAT64:
      typedData = new Float32Array(new Float64Array(imageData));
      break;
    default:
      typedData = new Float32Array(new Int16Array(imageData));
  }
  
  // Extract affine transformation
  const affine = [
    [header.affine[0][0], header.affine[0][1], header.affine[0][2], header.affine[0][3]],
    [header.affine[1][0], header.affine[1][1], header.affine[1][2], header.affine[1][3]],
    [header.affine[2][0], header.affine[2][1], header.affine[2][2], header.affine[2][3]],
    [0, 0, 0, 1],
  ];
  
  return { data: typedData, dims, affine };
}

/**
 * Load a NIfTI volume with caching
 */
export async function loadNiftiWithCache(
  sessionId: string,
  url: string
): Promise<NiftiVolume> {
  // Check cache first
  const cached = await getCachedVolume(sessionId);
  if (cached) {
    return {
      data: cached.data,
      dims: cached.dims,
      affine: [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]], // Default affine
    };
  }
  
  // Load from URL
  const volume = await loadNiftiFromUrl(url);
  
  // Cache for future use
  await cacheVolume(sessionId, volume.data, volume.dims);
  
  return volume;
}

/**
 * Extract a 2D slice from a 3D volume
 */
export function extractSlice(
  volume: NiftiVolume,
  view: 'sagittal' | 'coronal' | 'axial',
  sliceIndex: number
): Float32Array {
  const { data, dims } = volume;
  const [dimX, dimY, dimZ] = dims;
  
  let slice: Float32Array;
  
  switch (view) {
    case 'sagittal': {
      // X slice: dims are (Y, Z)
      const clampedIndex = Math.max(0, Math.min(sliceIndex, dimX - 1));
      slice = new Float32Array(dimY * dimZ);
      for (let z = 0; z < dimZ; z++) {
        for (let y = 0; y < dimY; y++) {
          slice[z * dimY + y] = data[clampedIndex + y * dimX + z * dimX * dimY];
        }
      }
      return slice;
    }
    
    case 'coronal': {
      // Y slice: dims are (X, Z)
      const clampedIndex = Math.max(0, Math.min(sliceIndex, dimY - 1));
      slice = new Float32Array(dimX * dimZ);
      for (let z = 0; z < dimZ; z++) {
        for (let x = 0; x < dimX; x++) {
          slice[z * dimX + x] = data[x + clampedIndex * dimX + z * dimX * dimY];
        }
      }
      return slice;
    }
    
    case 'axial': {
      // Z slice: dims are (X, Y)
      const clampedIndex = Math.max(0, Math.min(sliceIndex, dimZ - 1));
      slice = new Float32Array(dimX * dimY);
      for (let y = 0; y < dimY; y++) {
        for (let x = 0; x < dimX; x++) {
          slice[y * dimX + x] = data[x + y * dimX + clampedIndex * dimX * dimY];
        }
      }
      return slice;
    }
  }
}

/**
 * Get slice dimensions for a given view
 */
export function getSliceDims(
  dims: [number, number, number],
  view: 'sagittal' | 'coronal' | 'axial'
): [number, number] {
  const [dimX, dimY, dimZ] = dims;
  switch (view) {
    case 'sagittal':
      return [dimY, dimZ];
    case 'coronal':
      return [dimX, dimZ];
    case 'axial':
      return [dimX, dimY];
  }
}

/**
 * Get maximum slice index for a given view
 */
export function getMaxSliceIndex(
  dims: [number, number, number],
  view: 'sagittal' | 'coronal' | 'axial'
): number {
  const [dimX, dimY, dimZ] = dims;
  switch (view) {
    case 'sagittal':
      return dimX - 1;
    case 'coronal':
      return dimY - 1;
    case 'axial':
      return dimZ - 1;
  }
}
