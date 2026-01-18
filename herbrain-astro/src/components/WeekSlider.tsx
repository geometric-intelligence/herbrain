import { useState, useEffect } from 'react';

interface WeekSliderProps {
  value: number;
  onChange: (week: number) => void;
  min?: number;
  max?: number;
  label?: string;
}

export default function WeekSlider({
  value,
  onChange,
  min = 0,
  max = 40,
  label = 'Gestational Week',
}: WeekSliderProps) {
  const [localValue, setLocalValue] = useState(value);

  useEffect(() => {
    setLocalValue(value);
  }, [value]);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = parseInt(e.target.value, 10);
    setLocalValue(newValue);
  };

  const handleChangeEnd = () => {
    onChange(localValue);
  };

  return (
    <div className="w-full max-w-md">
      <div className="flex justify-between items-center mb-1.5">
        <label className="text-sm font-medium text-herbrain-dark">
          {label}
        </label>
        <span className="text-lg font-semibold text-herbrain-green bg-herbrain-green/10 px-2.5 py-0.5 rounded-lg">
          {localValue}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        value={localValue}
        onChange={handleChange}
        onMouseUp={handleChangeEnd}
        onTouchEnd={handleChangeEnd}
        className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-herbrain-green"
      />
      <div className="flex justify-between text-xs text-herbrain-muted mt-1">
        <span>Week {min}</span>
        <span>Week {max}</span>
      </div>
    </div>
  );
}
