import 'dotenv/config'; // Cara baru untuk mengimpor dotenv di ES Modules
import { GoogleGenAI } from '@google/genai';
import axios from 'axios';
import * as cheerio from 'cheerio';
import fs from 'fs';
import readline from 'readline-sync';

const API_KEY = process.env.GEMINI_API_KEY;
if (!API_KEY) {
    console.error('Error: GEMINI_API_KEY not found in .env file.');
    process.exit(1);
}

const ai = new GoogleGenAI({ apiKey: API_KEY });

const EMBEDDING_MODEL = 'gemini-embedding-exp-03-07';
const GENERATIVE_MODEL = 'gemini-2.0-flash';

const DATA_FILE = 'data-gemini.json';

// --- Helper Functions ---

async function getWebsiteContent(url) {
    console.log(`Mengambil konten dari: ${url}`);
    try {
        const response = await axios.get(url);
        const $ = cheerio.load(response.data);
        $('script, style, header, footer, nav, aside').remove();
        const textContent = $('body').text();
        return textContent.replace(/\s+/g, ' ').trim();
    } catch (error) {
        console.error(`Gagal mengambil konten dari URL ${url}:`, error.message);
        return null;
    }
}

function chunkText(text, chunkSize = 1000) {
    const chunks = [];
    const words = text.split(/\s+/);
    let currentChunk = [];
    for (const word of words) {
        currentChunk.push(word);
        if (currentChunk.length >= chunkSize) {
            chunks.push(currentChunk.join(' '));
            currentChunk = [];
        }
    }
    if (currentChunk.length > 0) {
        chunks.push(currentChunk.join(' '));
    }
    return chunks;
}

async function getEmbedding(text, taskType, retries = 5, delay = 3000) { // Menambahkan parameter retries dan delay
    for (let i = 0; i < retries; i++) {
        try {
            const response = await ai.models.embedContent({
                model: EMBEDDING_MODEL,
                contents: text,
                config: {
                    taskType: taskType
                }
            });

            const embeddings = response.embeddings?.[0]?.values;
            if (!embeddings) throw new Error('Embedding tidak tersedia dalam response.');

            return embeddings;

        } catch (error) {
            console.error(`Gagal mendapatkan embedding dengan model ${EMBEDDING_MODEL} (taskType: ${taskType}):`, error.message);
            if (error.response && error.response.data) {
                console.error(`Detail Error: ${JSON.stringify(error.response.data, null, 2)}`);
            } else {
                console.error(`Detail Error: ${error.message}`);
            }

            if (error.response && error.response.status === 429 && i < retries - 1) {
                console.log(`Menunggu ${delay / 1000} detik sebelum mencoba lagi...`);
                await new Promise(resolve => setTimeout(resolve, delay));
                delay *= 2; // Gandakan delay untuk percobaan berikutnya (exponential backoff)
            } else {
                return null; // Jika bukan 429 atau sudah mencapai batas retries, keluar
            }
        }
    }
    return null; // Jika semua percobaan gagal
}

function cosineSimilarityCorrected(vecA, vecB) {
    let dotProduct = 0;
    let magnitudeA = 0;
    let magnitudeB = 0;
    for (let i = 0; i < vecA.length; i++) {
        dotProduct += vecA[i] * vecB[i];
        magnitudeA += vecA[i] * vecA[i];
        magnitudeB += vecB[i] * vecB[i];
    }
    magnitudeA = Math.sqrt(magnitudeA);
    magnitudeB = Math.sqrt(magnitudeB);
    if (magnitudeA === 0 || magnitudeB === 0) return 0;
    return dotProduct / (magnitudeA * magnitudeB);
}

// --- Main Functions ---

async function trainFromWebsite(url) {
    const content = await getWebsiteContent(url);
    if (!content) {
        console.log('Tidak ada konten yang dapat diproses.');
        return;
    }

    const chunks = chunkText(content);
    console.log(`Website dipecah menjadi ${chunks.length} potongan (chunks).`);

    const data = [];
    for (let i = 0; i < chunks.length; i++) {
        process.stdout.write(`Membuat embedding untuk chunk ${i + 1}/${chunks.length}...`);
        const embedding = await getEmbedding(chunks[i], 'RETRIEVAL_DOCUMENT');
        if (embedding) {
            data.push({ text: chunks[i], embedding: embedding });
            process.stdout.write(' Selesai.\n');
        } else {
            process.stdout.write(' Gagal.\n');
        }
    }

    fs.writeFileSync(DATA_FILE, JSON.stringify(data, null, 2));
    console.log(`Training selesai. Data disimpan ke ${DATA_FILE}`);
}

async function trainFromMarkdown(filePath) {
    console.log(`Membaca file markdown dari: ${filePath}`);
    try {
        const content = fs.readFileSync(filePath, 'utf8');
        const chunks = chunkText(content);
        console.log(`File markdown dipecah menjadi ${chunks.length} potongan (chunks).`);

        const data = [];
        for (let i = 0; i < chunks.length; i++) {
            process.stdout.write(`Membuat embedding untuk chunk ${i + 1}/${chunks.length}...`);
            const embedding = await getEmbedding(chunks[i], 'RETRIEVAL_DOCUMENT');
            if (embedding) {
                data.push({ text: chunks[i], embedding: embedding });
                process.stdout.write(' Selesai.\n');
            } else {
                process.stdout.write(' Gagal.\n');
            }
        }

        fs.writeFileSync(DATA_FILE, JSON.stringify(data, null, 2));
        console.log(`Training selesai. Data disimpan ke ${DATA_FILE}`);
    } catch (error) {
        console.error(`Gagal membaca atau memproses file markdown:`, error.message);
    }
}

async function queryChatbot(query) {
    if (!fs.existsSync(DATA_FILE)) {
        console.log('Belum ada data training. Harap train dari website terlebih dahulu.');
        return;
    }

    const trainedData = JSON.parse(fs.readFileSync(DATA_FILE, 'utf8'));
    if (trainedData.length === 0) {
        console.log('Data training kosong. Harap train dari website terlebih dahulu.');
        return;
    }

    console.log('Mencari informasi relevan...');
    const queryEmbedding = await getEmbedding(query, 'RETRIEVAL_QUERY');
    if (!queryEmbedding) {
        console.log('Gagal membuat embedding untuk pertanyaan.');
        return;
    }

    let bestMatch = null;
    let maxSimilarity = -1;

    for (const item of trainedData) {
        // Menggunakan fungsi cosineSimilarityCorrected
        const similarity = cosineSimilarityCorrected(queryEmbedding, item.embedding);
        if (similarity > maxSimilarity) {
            maxSimilarity = similarity;
            bestMatch = item.text;
        }
    }

    let prompt;
    if (bestMatch && maxSimilarity > 0.7) {
        console.log(`Menemukan informasi relevan (kesamaan: ${maxSimilarity.toFixed(2)}).`);
        prompt = `Berdasarkan informasi berikut dari website:\n\n${bestMatch}\n\nJawablah pertanyaan ini: ${query}\nJika informasi tidak ada, katakan "Maaf, saya tidak menemukan informasi yang relevan di website yang Anda berikan."`;
    } else {
        console.log('Tidak menemukan informasi yang sangat relevan di website yang diberikan.');
        prompt = `Jawablah pertanyaan ini: ${query}\nJika Anda tidak memiliki informasi yang relevan, katakan "Maaf, saya tidak menemukan informasi yang relevan di website yang Anda berikan."`;
    }

    try {
        const response = await ai.models.generateContent({
            model: GENERATIVE_MODEL,
            contents: prompt,
            config: {
                responseMimeType: 'text/plain'
            }
        });

        console.log('\nGenerate Content:');
        console.log(response.text);
    } catch (error) {
        console.error(`Gagal mendapatkan jawaban dari Gemini dengan model ${GENERATIVE_MODEL}:`, error.message);
        if (error.response && error.response.data) {
            console.error(`Detail Error: ${JSON.stringify(error.response.data, null, 2)}`);
        } else if (error.message) {
            console.error(`Detail Error: ${error.message}`);
        }
        console.log('Pastikan pertanyaan Anda relevan dan/atau API Key Anda valid.');
    }
}

async function main() {
    console.log('--- AI Chatbot CLI ---');
    console.log('1. Train dari Website (Input URL)');
    console.log('2. Train dari File Markdown');
    console.log('3. Tanyakan Chatbot (Membutuhkan data training)');
    console.log('4. Keluar');

    let choice;
    while (choice !== '4') {
        choice = readline.question('Pilih opsi: ');

        switch (choice) {
            case '1':
                const url = readline.question('Masukkan URL website untuk training: ');
                if (url) {
                    await trainFromWebsite(url);
                } else {
                    console.log('URL tidak boleh kosong.');
                }
                break;
            case '2':
                const filePath = readline.question('Masukkan path file markdown untuk training: ');
                if (filePath) {
                    await trainFromMarkdown(filePath);
                } else {
                    console.log('Path file tidak boleh kosong.');
                }
                break;
            case '3':
                const query = readline.question('Tanyakan sesuatu kepada chatbot: ');
                if (query) {
                    await queryChatbot(query);
                } else {
                    console.log('Pertanyaan tidak boleh kosong.');
                }
                break;
            case '4':
                console.log('Sampai jumpa!');
                break;
            default:
                console.log('Pilihan tidak valid. Silakan coba lagi.');
        }
        console.log('\n---');
    }
}

main();