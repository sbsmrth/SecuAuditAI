import os
from dotenv import load_dotenv
from fastapi import FastAPI
from typing import Dict, Any

# 1. Cargar variables de entorno (asegúrate de tener el archivo .env en la misma carpeta)
load_dotenv()

# Componentes de LangChain
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Herramientas de Google Gemini
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

app = FastAPI(title="Motor de Auditoría RAG con Gemini")

# --- FASE 1: INDEXACIÓN ---
DIRECTORIO_DB = "./chroma_db"

# GoogleGenerativeAIEmbeddings buscará automáticamente 'GOOGLE_API_KEY' en el entorno
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-004")

def indexar_documentos():
    if not os.path.exists(DIRECTORIO_DB):
        print("Cargando manuales PDF desde /documents...")
        # Usamos PyPDFLoader para archivos PDF
        loader = DirectoryLoader('./documents', glob="**/*.pdf", loader_cls=PyPDFLoader)
        documentos_crudos = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        fragmentos = text_splitter.split_documents(documentos_crudos)
        
        Chroma.from_documents(documents=fragmentos, embedding=embeddings, persist_directory=DIRECTORIO_DB)
        print("Base de datos creada exitosamente.")

indexar_documentos()
vector_store = Chroma(persist_directory=DIRECTORIO_DB, embedding_function=embeddings)

# --- FASE 2: CONFIGURACIÓN DEL CEREBRO GEMINI ---
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0)

plantilla = """Eres un auditor de seguridad técnico.
Genera un análisis basado EXCLUSIVAMENTE en el siguiente contexto extraído de nuestros manuales:

[CONTEXTO RAG]
{contexto}

[HALLAZGO NMAP]
Puerto: {puerto}
Servicio: {servicio}

Redacta tu respuesta en dos párrafos cortos:
1. Riesgo Técnico (menciona CVEs y CVSS si el contexto los provee).
2. Acción de Mitigación requerida.
"""
prompt = ChatPromptTemplate.from_template(plantilla)

# --- FASE 3: EL ENDPOINT POST ---
@app.post("/analizar")
async def analizar_host(host: Dict[str, Any]):
    ip = host.get("ip", "Desconocida")
    puertos_abiertos = host.get("openPorts", [])
    
    if not puertos_abiertos:
        return {"resultado": f"🟢 IP {ip} sin puertos expuestos."}
    
    reporte_final = f"🚨 **REPORTE DE SEGURIDAD - IP: {ip}** 🚨\n\n"
    
    # Preparamos la cadena una vez
    cadena_rag = prompt | llm | StrOutputParser()
    
    for p in puertos_abiertos:
        puerto_id = str(p.get("port"))
        nombre_servicio = p.get("service")
        
        busqueda = f"Puerto {puerto_id} {nombre_servicio}"
        documentos_encontrados = vector_store.similarity_search(busqueda, k=1)
        
        contexto_extraido = documentos_encontrados[0].page_content if documentos_encontrados else "Sin documentación técnica disponible."
        
        respuesta_ia = cadena_rag.invoke({
            "contexto": contexto_extraido,
            "puerto": puerto_id,
            "servicio": nombre_servicio
        })
        
        reporte_final += f"🔹 **Puerto {puerto_id} ({nombre_servicio})**\n{respuesta_ia}\n\n"
        
    return {"resultado": reporte_final}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)