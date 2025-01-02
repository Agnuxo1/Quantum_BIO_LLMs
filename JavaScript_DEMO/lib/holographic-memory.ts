import { QuantumPhotonicProcessor, PhotonicState } from './quantum-photonic-processor';

interface MemoryCell {
  value: number;
  photonState: PhotonicState;
}

export class HolographicMemory {
  private memory: MemoryCell[][];
  private processor: QuantumPhotonicProcessor;
  private lastUpdateTime: number = 0;
  private readonly updateInterval: number = 50; // Reduced update frequency
  private processingChunk: boolean = false;

  constructor(private width: number, private height: number) {
    this.processor = new QuantumPhotonicProcessor({ width, height });
    this.initializeMemory();
  }

  private initializeMemory() {
    this.memory = Array(this.height).fill(0).map(() => 
      Array(this.width).fill(0).map(() => ({
        value: 0,
        photonState: this.processor.createPhotonState(0.1) // Reduced initial energy
      }))
    );
  }

  shouldUpdate(): boolean {
    const now = performance.now();
    if (now - this.lastUpdateTime >= this.updateInterval && !this.processingChunk) {
      this.lastUpdateTime = now;
      return true;
    }
    return false;
  }

  write(x: number, y: number, value: number): void {
    if (x < 0 || x >= this.width || y < 0 || y >= this.height) return;
    
    this.memory[y][x] = {
      value,
      photonState: this.processor.createPhotonState(value)
    };
  }

  async getMemoryState(): Promise<Float32Array> {
    const state = new Float32Array(this.width * this.height * 4);
    const chunkSize = 32; // Process in smaller chunks
    
    this.processingChunk = true;

    for (let y = 0; y < this.height; y += chunkSize) {
      const endY = Math.min(y + chunkSize, this.height);
      
      for (let x = 0; x < this.width; x += chunkSize) {
        const endX = Math.min(x + chunkSize, this.width);
        
        // Process chunk
        const interference = this.processor.generateInterferencePattern(
          { x, y, state: this.memory[y][x].photonState },
          { x: endX - 1, y: endY - 1, state: this.memory[endY - 1][endX - 1].photonState },
          { startX: x, startY: y, endX, endY }
        );
        
        // Copy chunk to state buffer
        for (let cy = y; cy < endY; cy++) {
          for (let cx = x; cx < endX; cx++) {
            const idx = (cy * this.width + cx) * 4;
            const interfIdx = ((cy - y) * (endX - x) + (cx - x));
            
            state[idx] = interference[interfIdx];
            state[idx + 1] = interference[interfIdx];
            state[idx + 2] = interference[interfIdx];
            state[idx + 3] = 1;
          }
        }
        
        // Allow other operations to process
        await new Promise(resolve => setTimeout(resolve, 0));
      }
    }
    
    this.processingChunk = false;
    return state;
  }
}

