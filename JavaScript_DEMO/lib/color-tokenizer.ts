export interface ColorToken {
  id: string;
  color: { r: number; g: number; b: number };
  embedding: number[];
  frequency: number;
}

export class ColorTokenizer {
  private tokens: Map<string, ColorToken>;
  private embeddingSize: number;

  constructor(embeddingSize: number = 64) {
    this.tokens = new Map();
    this.embeddingSize = embeddingSize;
  }

  // Generate a unique color ID based on RGB values
  private getColorId(r: number, g: number, b: number): string {
    const quantize = (v: number) => Math.round(v * 10) / 10;
    return `${quantize(r)}-${quantize(g)}-${quantize(b)}`;
  }

  // Generate a pseudo-random embedding vector for a color
  private generateEmbedding(r: number, g: number, b: number): number[] {
    const embedding = new Array(this.embeddingSize);
    for (let i = 0; i < this.embeddingSize; i++) {
      const phase = (i / this.embeddingSize) * Math.PI * 2;
      embedding[i] = (
        Math.sin(r * phase) +
        Math.sin(g * phase + Math.PI / 3) +
        Math.sin(b * phase + (2 * Math.PI) / 3)
      ) / 3;
    }
    return embedding;
  }

  // Tokenize a color into the vocabulary
  tokenize(r: number, g: number, b: number): ColorToken {
    const colorId = this.getColorId(r, g, b);
    
    if (this.tokens.has(colorId)) {
      const token = this.tokens.get(colorId)!;
      token.frequency += 1;
      return token;
    }

    const newToken: ColorToken = {
      id: colorId,
      color: { r, g, b },
      embedding: this.generateEmbedding(r, g, b),
      frequency: 1
    };

    this.tokens.set(colorId, newToken);
    return newToken;
  }

  // Find the most similar color in the vocabulary
  findSimilarColor(r: number, g: number, b: number): ColorToken | null {
    if (this.tokens.size === 0) return null;

    const targetEmbedding = this.generateEmbedding(r, g, b);
    let bestMatch: ColorToken | null = null;
    let bestSimilarity = -Infinity;

    for (const token of this.tokens.values()) {
      const similarity = this.cosineSimilarity(targetEmbedding, token.embedding);
      if (similarity > bestSimilarity) {
        bestSimilarity = similarity;
        bestMatch = token;
      }
    }

    return bestMatch;
  }

  // Calculate cosine similarity between two embeddings
  private cosineSimilarity(a: number[], b: number[]): number {
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;
    
    for (let i = 0; i < this.embeddingSize; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }
    
    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
  }

  // Get all tokens in the vocabulary
  getVocabulary(): ColorToken[] {
    return Array.from(this.tokens.values());
  }
}

