import { ColorToken, ColorTokenizer } from './color-tokenizer';

export class ColorRAGMemory {
  private memoryBuffer: Float32Array;
  private width: number;
  private height: number;
  private tokenizer: ColorTokenizer;
  private lastUpdate: number = 0;
  private updateThreshold: number = 16; // ms

  constructor(width: number, height: number) {
    this.width = width;
    this.height = height;
    this.memoryBuffer = new Float32Array(width * height * 4);
    this.tokenizer = new ColorTokenizer();
  }

  // Throttled update to prevent excessive state changes
  private shouldUpdate(): boolean {
    const now = performance.now();
    if (now - this.lastUpdate > this.updateThreshold) {
      this.lastUpdate = now;
      return true;
    }
    return false;
  }

  storePattern(pattern: { x: number; y: number; r: number; g: number; b: number; a: number }) {
    if (!this.shouldUpdate()) return;

    const index = (pattern.y * this.width + pattern.x) * 4;
    const token = this.tokenizer.tokenize(pattern.r, pattern.g, pattern.b);
    
    this.memoryBuffer[index] = token.color.r;
    this.memoryBuffer[index + 1] = token.color.g;
    this.memoryBuffer[index + 2] = token.color.b;
    this.memoryBuffer[index + 3] = pattern.a;
  }

  retrievePattern(x: number, y: number): { r: number; g: number; b: number; a: number } {
    const index = (y * this.width + x) * 4;
    return {
      r: this.memoryBuffer[index],
      g: this.memoryBuffer[index + 1],
      b: this.memoryBuffer[index + 2],
      a: this.memoryBuffer[index + 3],
    };
  }

  mixColors(color1: { r: number; g: number; b: number }, color2: { r: number; g: number; b: number }, ratio: number) {
    const token1 = this.tokenizer.tokenize(color1.r, color1.g, color1.b);
    const token2 = this.tokenizer.tokenize(color2.r, color2.g, color2.b);

    const mixedColor = {
      r: token1.color.r * (1 - ratio) + token2.color.r * ratio,
      g: token1.color.g * (1 - ratio) + token2.color.g * ratio,
      b: token1.color.b * (1 - ratio) + token2.color.b * ratio,
    };

    const similarToken = this.tokenizer.findSimilarColor(mixedColor.r, mixedColor.g, mixedColor.b);
    return similarToken ? similarToken.color : mixedColor;
  }

  applyInterference(x: number, y: number, radius: number, color: { r: number; g: number; b: number }) {
    if (!this.shouldUpdate()) return;

    for (let i = -radius; i <= radius; i++) {
      for (let j = -radius; j <= radius; j++) {
        const newX = x + i;
        const newY = y + j;
        
        if (newX >= 0 && newX < this.width && newY >= 0 && newY < this.height) {
          const distance = Math.sqrt(i * i + j * j);
          if (distance <= radius) {
            const intensity = (1 - distance / radius) * Math.cos(distance * Math.PI);
            const existingColor = this.retrievePattern(newX, newY);
            const mixedColor = this.mixColors(existingColor, color, Math.abs(intensity));
            this.storePattern({ x: newX, y: newY, ...mixedColor, a: 1 });
          }
        }
      }
    }
  }

  generateEntangledPattern(seed: number) {
    const pattern = new Float32Array(this.width * this.height * 4);
    for (let i = 0; i < this.width; i++) {
      for (let j = 0; j < this.height; j++) {
        const angle = (i + j + seed) * 0.1;
        const index = (j * this.width + i) * 4;
        
        const color = {
          r: (Math.sin(angle) + 1) * 0.5,
          g: (Math.cos(angle) + 1) * 0.5,
          b: (Math.sin(angle * 2) + 1) * 0.5
        };

        const token = this.tokenizer.tokenize(color.r, color.g, color.b);
        pattern[index] = token.color.r;
        pattern[index + 1] = token.color.g;
        pattern[index + 2] = token.color.b;
        pattern[index + 3] = 1;
      }
    }
    this.memoryBuffer = pattern;
  }

  getMemoryBuffer(): Float32Array {
    return this.memoryBuffer;
  }

  getTokenStats(): { totalTokens: number; frequencies: { [key: string]: number } } {
    const tokens = this.tokenizer.getVocabulary();
    const frequencies: { [key: string]: number } = {};
    
    tokens.forEach(token => {
      frequencies[token.id] = token.frequency;
    });

    return {
      totalTokens: tokens.length,
      frequencies
    };
  }
}

