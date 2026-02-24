import express from 'express';
import multer from 'multer';
import cors from 'cors';
import pdf from 'pdf-parse-fork';
import { InferenceClient } from "@huggingface/inference";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";

import { CONFIG } from './config.js';
import { supabase } from './supabaseClient.js';

const app = express();
const upload = multer({ storage: multer.memoryStorage() });
const hf = new InferenceClient(CONFIG.hfToken);

app.use(cors());
app.use(express.json());

// --- UTILITIES ---

const sanitizeText = (text) => {
  return text.replace(/\\/g, '\\\\').replace(/\u0000/g, '').replace(/[\u007F-\uFFFF]/g, (chr) => {
    return "\\u" + ("0000" + chr.charCodeAt(0).toString(16)).slice(-4);
  });
};

async function getEmbedding(text) {
  const response = await hf.featureExtraction({
    model: CONFIG.ai.embeddingModel,
    inputs: text,
  });
  return Array.isArray(response[0]) ? response[0] : response;
}

async function generateMatchScore(resumeText, jobDesc) {
  const prompt = `Analyze this resume against the Job Description. 
  Provide a match score from 0-100 based on technical fit.
  Return ONLY the numerical score.
  Job: ${jobDesc.substring(0, 500)}
  Resume: ${resumeText.substring(0, 2000)}`;

  const res = await hf.chatCompletion({
    model: CONFIG.ai.chatModel,
    messages: [{ role: "user", content: prompt }],
    max_tokens: 10,
    temperature: 0.1
  });
  return parseInt(res.choices[0].message.content.replace(/\D/g, '')) || 0;
}

// --- ENTRY POINTS ---

/**
 * 1. POST /upload-desc
 * Stores Job Description and its embedding
 */
app.post('/upload-desc', async (req, res) => {
  try {
    const { role, description, experience } = req.body;
    const embedding = await getEmbedding(`${role} ${description} ${experience}`);

    const { data, error } = await supabase
      .from('jobs')
      .insert({ role, description, experience, embedding })
      .select('id').single();

    if (error) throw error;
    res.json({ job_id: data.id });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

/**
 * 2. POST /ingest-resume
 * Extracts PDF, Chunks, Embeds, and Scores via SLM
 */
app.post('/ingest-resume', upload.single('file'), async (req, res) => {
  try {
    const { candidate_id, candidate_name, job_id } = req.body;
    
    // PDF Extraction & Chunking
    const pdfData = await pdf(req.file.buffer, {});
    const cleanText = sanitizeText(pdfData.text);
    
    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: CONFIG.chunks.size,
      chunkOverlap: CONFIG.chunks.overlap
    });
    const chunks = await splitter.splitText(cleanText);
    const fullContent = chunks.join(" ");

    // Fetch Job for context
    const { data: job } = await supabase.from('jobs').select('description').eq('id', job_id).single();
    
    const embedding = await getEmbedding(fullContent.substring(0, 3000));
    const score = await generateMatchScore(fullContent, job.description);

    const { error } = await supabase.from('resumes').insert({
      candidate_id,
      candidate_name,
      job_id,
      resume_text: cleanText,
      embedding,
      score
    });

    if (error) throw error;
    res.json({ message: "Resume processed", score });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

/**
 * 3. POST /candidate-list
 * Retrieves ranked candidates
 */
app.post('/candidate-list', async (req, res) => {
  try {
    const { job_id } = req.body;
    const { data, error } = await supabase
      .from('resumes')
      .select('candidate_id, candidate_name, score')
      .eq('job_id', job_id)
      .order('score', { ascending: false });

    if (error) throw error;
    res.json(data);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

/**
 * 4. GET /get-jobids
 * Returns all job IDs and roles
 */
app.get('/get-jobids', async (req, res) => {
  try {
    const { data, error } = await supabase.from('jobs').select('id, role');
    if (error) throw error;
    res.json(data);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.listen(CONFIG.port, () => console.log(`🚀 Resume Parser Server on port ${CONFIG.port}`));