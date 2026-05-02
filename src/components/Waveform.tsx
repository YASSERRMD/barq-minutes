import { useEffect, useRef } from 'react';

export default function Waveform({ analyser }: { analyser: AnalyserNode | null }) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !analyser) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const data = new Uint8Array(analyser.fftSize);
    let frame = 0;

    const draw = () => {
      frame = requestAnimationFrame(draw);
      analyser.getByteTimeDomainData(data);

      const width = canvas.width;
      const height = canvas.height;
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = 'hsl(150 12% 96%)';
      ctx.fillRect(0, 0, width, height);
      ctx.strokeStyle = 'hsl(150 60% 40%)';
      ctx.lineWidth = 2;
      ctx.beginPath();

      const slice = width / data.length;
      for (let i = 0; i < data.length; i++) {
        const x = i * slice;
        const y = (data[i] / 255) * height;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    };

    draw();
    return () => cancelAnimationFrame(frame);
  }, [analyser]);

  return <canvas ref={canvasRef} className="waveform" width={960} height={220} aria-label="Live waveform" />;
}
