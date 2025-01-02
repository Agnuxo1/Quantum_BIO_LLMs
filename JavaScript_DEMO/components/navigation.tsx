import Link from 'next/link'
import { Button } from '@/components/ui/button'
import { ModeToggle } from '@/components/mode-toggle'
import { Brain, Activity, Zap, Home } from 'lucide-react'

export function Navigation() {
  return (
    <header className="border-b">
      <div className="container mx-auto flex h-16 items-center justify-between">
        <div className="flex items-center gap-6">
          <Link href="/" className="flex items-center gap-2 font-bold">
            <Brain className="h-6 w-6" />
            <span>Quantum BIO LLMs</span>
          </Link>
          <nav className="flex gap-4">
            <Link href="/dashboard">
              <Button variant="ghost">
                <Activity className="mr-2 h-4 w-4" />
                Dashboard
              </Button>
            </Link>
            <Link href="/demos">
              <Button variant="ghost">
                <Zap className="mr-2 h-4 w-4" />
                Demos
              </Button>
            </Link>
          </nav>
        </div>
        <ModeToggle />
      </div>
    </header>
  )
}

