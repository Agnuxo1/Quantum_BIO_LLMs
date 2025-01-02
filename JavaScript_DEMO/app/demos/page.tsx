import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Brain, Cpu, Zap, MessageSquare } from 'lucide-react'
import Link from 'next/link'

export default function Demos() {
  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold">Interactive Demos</h1>
      <p className="text-muted-foreground">
        Explore the capabilities of our Quantum BIO LLM system through these interactive demonstrations.
      </p>

      <div className="grid md:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Cpu className="h-5 w-5" />
              Quantum Processing Demo
            </CardTitle>
            <CardDescription>
              Visualize quantum optimization in action
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <p>
              Experience how our quantum-inspired algorithms optimize language model processing in real-time.
            </p>
            <Link href="/demos/quantum">
              <Button>
                Launch Demo
              </Button>
            </Link>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Brain className="h-5 w-5" />
              Neural Architecture Demo
            </CardTitle>
            <CardDescription>
              Explore the bioinspired memory system
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <p>
              See how our biologically inspired architecture handles information processing and storage.
            </p>
            <Link href="/demos/neural">
              <Button>
                Launch Demo
              </Button>
            </Link>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Zap className="h-5 w-5" />
              Ray Tracing Visualization
            </CardTitle>
            <CardDescription>
              Interactive data visualization demo
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <p>
              Interact with our ray tracing system for enhanced data visualization and processing.
            </p>
            <Link href="/demos/raytracing">
              <Button>
                Launch Demo
              </Button>
            </Link>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <MessageSquare className="h-5 w-5" />
              LLM Chat with Visualization
            </CardTitle>
            <CardDescription>
              Chat with the LLM and visualize the underlying processes
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <p>
              Interact with our LLM while visualizing the tokenization, neural network, and holographic memory.
            </p>
            <Link href="/demos/llm-chat">
              <Button>
                Launch Demo
              </Button>
            </Link>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

