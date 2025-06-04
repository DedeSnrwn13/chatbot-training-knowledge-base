const axios = require('axios');
const cheerio = require('cheerio');
const readline = require('readline-sync');
const fs = require('fs');
const { OpenAI } = require('openai');

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY }); // Ganti dengan API key milikmu

async function scrapeWebsite(url) {
    try {
        const response = await axios.get(url);
        const $ = cheerio.load(response.data);
        return $('body').text().replace(/\s+/g, ' ').trim().slice(0, 3000); // batasi token
    } catch (err) {
        console.error('Gagal mengambil konten website:', err.message);
        return null;
    }
}

async function generateEmbedding(text) {
    const res = await openai.embeddings.create({
        model: 'text-embedding-3-small',
        input: text
    });
    return res.data[0].embedding;
}

function saveEmbedding(embedding, text) {
    const data = { text, embedding };
    fs.writeFileSync('embedding.json', JSON.stringify(data));
    console.log('Embedding disimpan ke file embedding.json');
}

function cosineSimilarity(a, b) {
    const dot = a.reduce((sum, ai, i) => sum + ai * b[i], 0);
    const normA = Math.sqrt(a.reduce((sum, ai) => sum + ai * ai, 0));
    const normB = Math.sqrt(b.reduce((sum, bi) => sum + bi * bi, 0));
    return dot / (normA * normB);
}

async function answerQuestion(question) {
    const file = JSON.parse(fs.readFileSync('embedding.json'));
    const userEmbedding = await generateEmbedding(question);
    const similarity = cosineSimilarity(userEmbedding, file.embedding);

    const context = similarity > 0.7 ? file.text : ''; // threshold relevansi

    const response = await openai.chat.completions.create({
        model: 'gpt-4',
        messages: [
            { role: 'system', content: 'Kamu adalah asisten berbasis dokumen website.' },
            { role: 'user', content: `Jawab berdasarkan konten ini:\n${context}\n\nPertanyaan: ${question}` }
        ],
        temperature: 0.2,
    });

    console.log('\nJawaban AI:', response.choices[0].message.content.trim());
}

(async () => {
    const mode = readline.question('Mode (train/ask): ');

    if (mode === 'train') {
        const url = readline.question('Masukkan URL website: ');
        const content = await scrapeWebsite(url);
        if (!content) return;

        const embedding = await generateEmbedding(content);
        saveEmbedding(embedding, content);
    } else if (mode === 'ask') {
        const question = readline.question('Tanya AI: ');
        await answerQuestion(question);
    } else {
        console.log('Mode tidak dikenali. Gunakan "train" atau "ask".');
    }
})();
