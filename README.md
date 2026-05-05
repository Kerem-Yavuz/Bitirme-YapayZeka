# 🎓 Ders Seçim Chatbot — RAG Engine

Üniversite ders seçim asistanı. Dokümanlardan (PDF, TXT, MD) otomatik bilgi çıkarır, vektör veritabanında saklar ve öğrenci sorularına bağlam destekli yanıt üretir.

## Mimari

```
Kullanıcı → FastAPI Server → Semantic Router → Easy/Hard LLM
                                    ↓
                           RAG Engine (Qdrant)
                           ├── ders_docs_v2      (doküman vektörleri)
                           ├── _embedding_cache   (embedding önbelleği)
                           └── _system_config     (sistem ayarları)
```

**Temel Prensipler:**
- **Zero local file dependency** — tüm veri Qdrant'ta
- **Semantic routing** — sorular otomatik olarak Easy LLM (basit) / Hard LLM (karmaşık) / Reject'e yönlenir
- **Gerçek zamanlı streaming** — LLM'den gelen tokenlar anında istemciye iletilir
- **Incremental ingest** — yalnızca değişen dokümanlar yeniden işlenir

## Teknolojiler

| Bileşen | Teknoloji |
|---------|-----------|
| API Server | **FastAPI** + Uvicorn (ASGI) |
| Vektör DB | Qdrant |
| Embedding | sentence-transformers/all-MiniLM-L6-v2 |
| LLM | llama.cpp (OpenAI-compatible API) |
| Router | semantic-router |
| Doküman İşleme | pypdf, NLTK |

## Kurulum

### 1. Bağımlılıkları yükle

```bash
pip install -r requirements.txt
```

### 2. Ortam değişkenlerini ayarla

```bash
cp .env.example .env
# .env dosyasını düzenle — özellikle RESET_API_KEY'i değiştir!
```

### 3. Servislerin çalıştığını kontrol et

- **Qdrant** → `QDRANT_URL` adresinde erişilebilir olmalı
- **llama.cpp** → `LLAMA_CPP_URL` adresinde erişilebilir olmalı

### 4. Dokümanları ingest et

```bash
# CLI ile
python rag_qdrant.py ingest ./data

# Veya API üzerinden
curl -X POST http://localhost:5000/api/ingest \
  -H "Content-Type: application/json" \
  -d '{"folder": "./data"}'
```

### 5. Sunucuyu başlat

```bash
# Development (auto-reload)
python qdrant_gui.py --reload

# Production
uvicorn qdrant_gui:app --host 0.0.0.0 --port 5000 --workers 4 --timeout-keep-alive 300
```

### 6. API Dokümantasyonu

Sunucu başlatıldıktan sonra otomatik Swagger arayüzü:
- **Swagger UI:** http://localhost:5000/docs
- **ReDoc:** http://localhost:5000/redoc

## Docker

```bash
docker compose up --build
```

## API Endpoints

| Endpoint | Metod | Açıklama |
|----------|-------|----------|
| `/` | GET | Web GUI |
| `/api/ask` | POST | Soru sor (otomatik routing, gerçek zamanlı streaming) |
| `/api/search` | POST | Vektör arama (LLM'siz) |
| `/api/ingest` | POST | Doküman ingest |
| `/api/info` | GET | Collection bilgisi |
| `/api/health` | GET | Sağlık kontrolü |
| `/api/qdrant/reset` | POST | Collection sıfırla (🔒 admin key gerekli) |
| `/docs` | GET | Swagger UI (otomatik) |

### Örnek: Soru Sorma

```bash
curl -X POST http://localhost:5000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Bu dersin kredisi kaç?"}'
```

### Örnek: Admin Reset

```bash
curl -X POST http://localhost:5000/api/qdrant/reset \
  -H "X-Reset-Key: BURAYA_KENDI_KEYINIZI_YAZIN"
```

## CLI Komutları

```bash
# Doküman ingest
python rag_qdrant.py ingest ./data [--force]

# Vektör arama
python rag_qdrant.py search "müfredat" [--top-k 10] [--json]

# Soru sor (LLM ile)
python rag_qdrant.py ask "Bu dersin ön koşulu ne?" [--json]

# Collection bilgisi
python rag_qdrant.py info

# Router ile soru sor
python router.py ask "Hangi seçmeli dersleri almalıyım?"
```

## Desteklenen Dosya Formatları

- `.pdf` — PDF dokümanları
- `.txt` — Düz metin
- `.md` — Markdown

## Proje Yapısı

```
├── qdrant_gui.py      # FastAPI server: async endpoints, real-time streaming
├── rag_qdrant.py      # RAG engine: chunking, embedding, arama, LLM
├── router.py          # Semantic router: easy/hard/reject yönlendirme
├── config.py          # Merkezi konfigürasyon
├── rag_gui.html       # Web GUI (tek dosya)
├── requirements.txt   # Python bağımlılıkları
├── Dockerfile         # Container build
├── docker-compose.yml # Docker Compose
├── setup.sh           # Kurulum script'i
└── .env.example       # Ortam değişkenleri şablonu
```
