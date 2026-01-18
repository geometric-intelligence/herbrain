/**
 * IndexedDB cache helpers for MRI volumes
 */
import { get, set, del, keys } from 'idb-keyval';

const CACHE_PREFIX = 'herbrain-mri-';
const MAX_CACHE_SIZE = 10; // Maximum number of cached volumes

export interface CachedVolume {
  data: Float32Array;
  dims: [number, number, number];
  timestamp: number;
}

/**
 * Get a cached MRI volume by session ID
 */
export async function getCachedVolume(sessionId: string): Promise<CachedVolume | null> {
  try {
    const cached = await get<CachedVolume>(`${CACHE_PREFIX}${sessionId}`);
    return cached || null;
  } catch (error) {
    console.warn('Failed to get cached volume:', error);
    return null;
  }
}

/**
 * Cache an MRI volume
 */
export async function cacheVolume(
  sessionId: string,
  data: Float32Array,
  dims: [number, number, number]
): Promise<void> {
  try {
    // Clean up old entries if cache is full
    const allKeys = await keys();
    const cacheKeys = allKeys.filter(k => String(k).startsWith(CACHE_PREFIX));
    
    if (cacheKeys.length >= MAX_CACHE_SIZE) {
      // Remove oldest entry
      const oldest = cacheKeys[0];
      await del(oldest);
    }
    
    await set(`${CACHE_PREFIX}${sessionId}`, {
      data,
      dims,
      timestamp: Date.now(),
    });
  } catch (error) {
    console.warn('Failed to cache volume:', error);
  }
}

/**
 * Clear all cached volumes
 */
export async function clearCache(): Promise<void> {
  try {
    const allKeys = await keys();
    const cacheKeys = allKeys.filter(k => String(k).startsWith(CACHE_PREFIX));
    await Promise.all(cacheKeys.map(k => del(k)));
  } catch (error) {
    console.warn('Failed to clear cache:', error);
  }
}

/**
 * Store OpenAI API key in localStorage
 */
export function getApiKey(): string | null {
  if (typeof window === 'undefined') return null;
  return localStorage.getItem('herbrain-openai-key');
}

export function setApiKey(key: string): void {
  if (typeof window === 'undefined') return;
  localStorage.setItem('herbrain-openai-key', key);
}

export function clearApiKey(): void {
  if (typeof window === 'undefined') return;
  localStorage.removeItem('herbrain-openai-key');
}
