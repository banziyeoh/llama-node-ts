import { LLM } from "llama-node";
import { LLamaCpp } from "llama-node/dist/llm/llama-cpp.js";
import path from "path";
import * as readline from "readline";

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

const model = path.resolve(process.cwd(), "./models/ggml-vic7b-q5_1.bin");
const llama = new LLM(LLamaCpp);
const config: any = {
  path: model,
  enableLogging: true,
  nCtx: 1024,
  nParts: -1,
  seed: 0,
  f16Kv: false,
  logitsAll: false,
  vocabOnly: false,
  useMlock: false,
  embedding: false,
  useMmap: true,
  nGpuLayers: 0,
};

const template = `How are you?`;
// const prompt = `A chat between a user and an assistant.
// USER: ${template}
// ASSISTANT:`;

const seedPrompt = [];
const userPrompts: string[] = [];
const assistantResponses: string[] = [];

await llama.load(config);

// const run = async () => {

// }

// let currentPrompt = ""
// while (currentPrompt.toLowerCase() != "stop") {
//     rl.question(`Prompt > `, (prompt: string) => {
//     currentPrompt = prompt;
//       userPrompts.push(prompt);
//     });
//     // //construct prompt
//     // let prompt = "";
// for (let i = 0; i < userPrompts.length; i++) {
//   prompt += `USER: ${userPrompts[i]}\nASSISTANT: ${assistantResponses[i]}\n`;
// }

// //generate response
// await llama.createCompletion(
//   {
//     nThreads: 4,
//     nTokPredict: 2048,
//     topK: 40,
//     topP: 0.1,
//     temp: 0.2,
//     repeatPenalty: 1,
//     prompt: prompt,
//   },
//   (response) => {
//     process.stdout.write(response.token);
//   }
// );

// }
// rl.close();
// //run();

let more = 1;
read();

let prompt = "A chat between a user and an assistant.\n";

async function read() {
  rl.question("Prompt > ", async function (answer) {
    if (answer == "no") {
      more = 0;
      console.log("bye");
      rl.close();
    } else {
      more++;
      prompt += `USER: ${answer}\n`;
      //userPrompts.push(answer);

      //   for (let i = 0; i < userPrompts.length; i++) {
      //     if (i == assistantResponses.length - 1) {
      //       prompt += `USER: ${userPrompts[i]}\nASSISTANT: ${assistantResponses[i]}\n`;
      //       break;
      //     }
      //     prompt += `USER: ${userPrompts[i]}\nASSISTANT: ${assistantResponses[i]}\n`;
      //   }
      let assistantResponse = "";
      await llama.createCompletion(
        {
          nThreads: 4,
          nTokPredict: 2048,
          topK: 40,
          topP: 0.1,
          temp: 0.2,
          repeatPenalty: 1,
          prompt: prompt,
        },
        (response) => {
        if (response.token.includes("<end>")){
            console.log("end of sentence");
            return
        }
          assistantResponse = assistantResponse.concat(response.token);
          process.stdout.write(response.token);
        }
      );
      prompt += `ASSISTANT: ${assistantResponse}\n`;

      console.log("Current Conversation: \n" + prompt);
      console.log("next round.." + more);
      read();
    }
  });
}
