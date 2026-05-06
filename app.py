from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from collections import Counter
import numpy as np
import uvicorn
import re

app = FastAPI()

class TrendRequest(BaseModel):
    texts: List[str]
    time_periods: Optional[List[str]] = None

class TrendResponse(BaseModel):
    status: str
    trending_up: List[str]
    trending_down: List[str]
    frequency_change: List[dict]

# Palabras comunes a ignorar (stopwords en español)
STOPWORDS = {
    'de', 'la', 'que', 'el', 'en', 'y', 'a', 'los', 'del', 'las', 'un', 'por', 'con', 'no',
    'una', 'es', 'para', 'como', 'su', 'al', 'lo', 'más', 'pero', 'sus', 'le', 'ya', 'cuando',
    'este', 'esta', 'ese', 'esa', 'fue', 'han', 'sido', 'tan', 'muy', 'sin', 'sobre', 'tras',
    'ante', 'ante', 'cada', 'todo', 'toda', 'todos', 'todas', 'uno', 'una', 'unos', 'unas'
}

def clean_text(text: str) -> str:
    """Limpia el texto: minúsculas, elimina puntuación, normaliza"""
    text = text.lower()
    # Eliminar puntuación
    text = re.sub(r'[^\w\s]', ' ', text)
    # Eliminar números
    text = re.sub(r'\d+', ' ', text)
    # Eliminar espacios múltiples
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def get_word_frequency(texts: List[str]) -> Counter:
    """Extrae frecuencia de palabras significativas"""
    word_counter = Counter()
    
    for text in texts:
        cleaned = clean_text(text)
        words = cleaned.split()
        
        for word in words:
            # Filtrar palabras cortas y stopwords
            if len(word) > 3 and word not in STOPWORDS:
                word_counter[word] += 1
    
    return word_counter

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/")
async def detect_trends(req: TrendRequest):
    if not req.texts or len(req.texts) < 4:
        raise HTTPException(status_code=400, detail="Se necesitan al menos 4 textos para detectar tendencias")
    
    # Dividir en dos mitades (antes/después)
    mitad = len(req.texts) // 2
    primera_mitad = req.texts[:mitad]
    segunda_mitad = req.texts[mitad:]
    
    # Calcular frecuencias
    freq1 = get_word_frequency(primera_mitad)
    freq2 = get_word_frequency(segunda_mitad)
    
    total1 = sum(freq1.values()) if freq1 else 1
    total2 = sum(freq2.values()) if freq2 else 1
    
    # Calcular cambios
    todas_palabras = set(freq1.keys()) | set(freq2.keys())
    cambios = []
    
    for palabra in todas_palabras:
        count1 = freq1.get(palabra, 0)
        count2 = freq2.get(palabra, 0)
        
        # Frecuencia normalizada por total de palabras
        norm1 = count1 / total1
        norm2 = count2 / total2
        
        if norm1 > 0 or norm2 > 0:
            # Cambio porcentual (evitando división por cero)
            if norm1 > 0:
                cambio_pct = ((norm2 - norm1) / norm1) * 100
            else:
                cambio_pct = 100 if norm2 > 0 else 0
            
            cambios.append({
                "term": palabra,
                "current_freq": round(norm2, 6),
                "past_freq": round(norm1, 6),
                "change_percent": round(cambio_pct, 1)
            })
    
    # Ordenar por cambio porcentual (mayor a menor)
    cambios.sort(key=lambda x: x["change_percent"], reverse=True)
    
    # Palabras en tendencia al alza (más del 30% de aumento)
    trending_up = []
    # Palabras en tendencia a la baja (más del 30% de disminución)
    trending_down = []
    
    for c in cambios:
        if c["change_percent"] > 30 and c["current_freq"] > 0.001:
            trending_up.append(c["term"])
        elif c["change_percent"] < -30 and c["past_freq"] > 0.001:
            trending_down.append(c["term"])
    
    # Limitar a 10 términos por categoría
    trending_up = trending_up[:10]
    trending_down = trending_down[:10]
    
    return {
        "status": "ok",
        "trending_up": trending_up,
        "trending_down": trending_down,
        "frequency_change": cambios[:20]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
