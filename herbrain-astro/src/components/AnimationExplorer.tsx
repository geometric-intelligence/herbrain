import { useRef, useEffect } from 'react';

interface AnimationExplorerProps {
  week: number;
}

// Fetus size comparisons by gestational week (common pregnancy milestone descriptions)
const FETUS_SIZE_BY_WEEK: Record<number, string> = {
  0: 'a poppy seed',
  1: 'a poppy seed',
  2: 'a poppy seed',
  3: 'a poppy seed',
  4: 'a poppy seed',
  5: 'a sesame seed',
  6: 'a lentil',
  7: 'a blueberry',
  8: 'a raspberry',
  9: 'a grape',
  10: 'a kumquat',
  11: 'a fig',
  12: 'a lime',
  13: 'a lemon',
  14: 'a peach',
  15: 'an apple',
  16: 'an avocado',
  17: 'a pear',
  18: 'a bell pepper',
  19: 'a mango',
  20: 'a banana',
  21: 'a carrot',
  22: 'a papaya',
  23: 'a grapefruit',
  24: 'an ear of corn',
  25: 'a cauliflower',
  26: 'a lettuce head',
  27: 'a cabbage',
  28: 'an eggplant',
  29: 'a butternut squash',
  30: 'a coconut',
  31: 'a pineapple',
  32: 'a squash',
  33: 'a durian',
  34: 'a cantaloupe',
  35: 'a honeydew melon',
  36: 'a romaine lettuce',
  37: 'a winter melon',
  38: 'a leek',
  39: 'a mini watermelon',
  40: 'a small pumpkin',
};

function getFetusSize(week: number): string {
  // Clamp week to valid range
  const clampedWeek = Math.max(0, Math.min(40, Math.round(week)));
  return FETUS_SIZE_BY_WEEK[clampedWeek] || 'a small pumpkin';
}

/**
 * AnimationExplorer component that displays a pregnancy animation video
 * with frame seeking based on gestational week.
 * 
 * The video has 10 frames (weeks 00, 05, 10, 15, 20, 25, 30, 35, 40, 41) at 1fps.
 * Frame seeking happens entirely in the browser - no network requests.
 */
export default function AnimationExplorer({ week }: AnimationExplorerProps) {
  const videoRef = useRef<HTMLVideoElement>(null);

  // Map gestational week to video time (frame number at 1fps)
  const getFrameTime = (gestWeek: number): number => {
    if (gestWeek >= 41) {
      return 9;
    } else if (gestWeek >= 40) {
      return 8;
    } else {
      return Math.floor(gestWeek / 5);
    }
  };

  // Seek video to correct frame when week changes
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const frameTime = getFrameTime(week);
    // Add small offset to ensure we're in the frame
    video.currentTime = frameTime + 0.001;
  }, [week]);

  // Initial load handler
  const handleLoadedData = () => {
    const video = videoRef.current;
    if (!video) return;

    // Seek to initial frame
    const frameTime = getFrameTime(week);
    video.currentTime = frameTime + 0.001;
  };

  return (
    <div className="flex flex-col items-center justify-center h-full">
      <div className="bg-white rounded-xl border border-gray-200 p-3 overflow-hidden">
        <video
          ref={videoRef}
          src="/assets/pregnancy_animation.mp4"
          preload="auto"
          muted
          playsInline
          onLoadedData={handleLoadedData}
          className="max-h-72 w-auto object-contain"
          style={{ display: 'block', margin: '0 auto' }}
        />
        
        {/* Fetus size description */}
        <p className="text-xs text-center text-herbrain-muted mt-2 px-2">
          The fetus is about the size of {getFetusSize(week)}
        </p>
      </div>
    </div>
  );
}
