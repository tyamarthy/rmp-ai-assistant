import { NextResponse } from 'next/server';
import { Pinecone } from '@pinecone-database/pinecone';
import * as genai from 'google.generativeai';
import { OpenAI } from 'openai';


const pc = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY,
});


const index = pc.index('rag').namespace('ns1');


const aiClient = genai.configure({
  apiKey: process.env.GOOGLE_AI_API_KEY,
});


const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});


const systemPrompt = `
You are a rate my professor agent to help students find classes. You take in user questions and answer them.
For every user question, the top 3 professors that match the user question are returned.
Use them to answer the question if needed.
`;


async function createEmbedding(text) {
  try {
    const response = await openai.embeddings.create({
      model: 'text-embedding-3-small',
      input: text,
    });
    return response.data.embeddings[0]; 
  } catch (error) {
    console.error('Error creating embedding:', error);
    throw error;
  }
}

export async function POST(request) {
  try {

    const { userQuestion } = await request.json();

  
    const questionEmbedding = await createEmbedding(userQuestion);

 
    const searchResults = await index.query({
      vector: questionEmbedding,
      topK: 5,
      includeMetadata: true,
    });

    let resultString = '';
    searchResults.matches.forEach((match) => {
      resultString += `
      Returned Results:
      Professor: ${match.id}
      Review: ${match.metadata.review}
      Subject: ${match.metadata.subject}
      Stars: ${match.metadata.stars}
      \n\n`;
    });

    
    const lastMessageContent = `${userQuestion}\n\n${resultString}`;
    const prompt = `${systemPrompt}\nUser Question: ${userQuestion}\nTop Professors: ${resultString}`;

   
    const response = await aiClient.text({
      prompt: prompt,
    });


    return NextResponse.json({ answer: response.result });
  } catch (error) {
    console.error('Error processing request:', error);
    return NextResponse.error();
  }
}
