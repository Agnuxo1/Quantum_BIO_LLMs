import tkinter as tk
from tkinter import scrolledtext, ttk
import json
from datetime import datetime
from typing import Dict, Any, Optional
import threading

class ChatInterface:
    def __init__(self, model: Any):
        """
        Interfaz de chat para el modelo cuántico-bioinspirado.
        
        Args:
            model: Modelo a usar para las respuestas
        """
        self.model = model
        self.window = None
        self.chat_history = None
        self.user_input = None
        self.send_button = None
        self.status_bar = None
        self.processing = False
        self.history: List[Dict[str, str]] = []
        
    def initialize_window(self):
        """Inicializa la ventana principal."""
        self.window = tk.Tk()
        self.window.title("Chat Cuántico-Bioinspirado")
        self.window.geometry("800x600")
        
        # Área de chat
        self.chat_history = scrolledtext.ScrolledText(
            self.window,
            wrap=tk.WORD,
            width=70,
            height=30
        )
        self.chat_history.pack(padx=10, pady=10, expand=True, fill='both')
        
        # Frame para entrada y botón
        input_frame = ttk.Frame(self.window)
        input_frame.pack(fill='x', padx=10, pady=5)
        
        # Campo de entrada
        self.user_input = ttk.Entry(input_frame)
        self.user_input.pack(side='left', fill='x', expand=True)
        self.user_input.bind("<Return>", lambda e: self.send_message())
        
        # Botón de envío
        self.send_button = ttk.Button(
            input_frame,
            text="Enviar",
            command=self.send_message
        )
        self.send_button.pack(side='right', padx=5)
        
        # Barra de estado
        self.status_bar = ttk.Label(
            self.window,
            text="Listo",
            anchor='w'
        )
        self.status_bar.pack(fill='x', padx=10, pady=5)
        
    def send_message(self):
        """Procesa y envía el mensaje del usuario."""
        if self.processing:
            return
            
        message = self.user_input.get().strip()
        if not message:
            return
            
        # Limpiar entrada
        self.user_input.delete(0, tk.END)
        
        # Mostrar mensaje del usuario
        self.append_message("Usuario", message)
        
        # Procesar respuesta en thread separado
        self.processing = True
        self.update_status("Procesando...")
        threading.Thread(target=self.process_response, args=(message,)).start()
        
    def process_response(self, message: str):
        """
        Procesa la respuesta del modelo.
        
        Args:
            message: Mensaje del usuario
        """
        try:
            response = self.model.generate_response(message)
            self.window.after(0, self.append_message, "AI", response)
        except Exception as e:
            self.window.after(0, self.append_message, "Error", str(e))
        finally:
            self.processing = False
            self.window.after(0, self.update_status, "Listo")
            
    def append_message(self, sender: str, message: str):
        """
        Añade un mensaje al historial.
        
        Args:
            sender: Remitente del mensaje
            message: Contenido del mensaje
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Añadir al historial
        self.history.append({
            'timestamp': timestamp,
            'sender': sender,
            'message': message
        })
        
        # Mostrar en interfaz
        self.chat_history.insert(tk.END, f"[{timestamp}] {sender}: {message}\n\n")
        self.chat_history.see(tk.END)
        
    def update_status(self, status: str):
        """
        Actualiza la barra de estado.
        
        Args:
            status: Nuevo estado
        """
        self.status_bar.config(text=status)
        
    def save_history(self, filepath: str):
        """
        Guarda el historial de chat.
        
        Args:
            filepath: Ruta donde guardar el historial
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)
            
    def load_history(self, filepath: str):
        """
        Carga historial de chat.
        
        Args:
            filepath: Ruta del historial a cargar
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                self.history = json.load(f)
                
            # Mostrar historial cargado
            self.chat_history.delete('1.0', tk.END)
            for msg in self.history:
                self.chat_history.insert(
                    tk.END,
                    f"[{msg['timestamp']}] {msg['sender']}: {msg['message']}\n\n"
                )
        except Exception as e:
            self.append_message("Sistema", f"Error cargando historial: {str(e)}")
            
    def run(self):
        """Inicia la interfaz."""
        if self.window is None:
            self.initialize_window()
        self.window.mainloop()
