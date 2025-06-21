import { Anthropic } from "@anthropic-ai/sdk"
import OpenAI from "openai"
import { ApiHandler } from "../"
import {
	ApiHandlerOptions,
	HuaweiCloudMaasModelId,
	ModelInfo,
	huaweiCloudMaasDefaultModelId,
	huaweiCloudMaasModels,
} from "@shared/api"
import { convertToOpenAiMessages } from "@api/transform/openai-format"
import { ApiStream } from "@api/transform/stream"
import { withRetry } from "../retry"

export class HuaweiCloudMaasHandler implements ApiHandler {
	private options: ApiHandlerOptions
	private client: OpenAI

	constructor(options: ApiHandlerOptions) {
		this.options = options
		this.client = new OpenAI({
			baseURL: "https://api.modelarts-maas.com/v1",
			apiKey: this.options.huaweiCloudApiKey,
		})
	}

	@withRetry()
	async *createMessage(systemPrompt: string, messages: Anthropic.Messages.MessageParam[]): ApiStream {
		const modelId = this.getModel().id

		const stream = await this.client.chat.completions.create({
			model: modelId,
			messages: [{ role: "system", content: systemPrompt }, ...convertToOpenAiMessages(messages)],
			stream: true,
		})

		for await (const chunk of stream) {
			const delta = chunk.choices[0]?.delta
			if (delta?.content) {
				yield {
					type: "text",
					text: delta.content,
				}
			}

			if (delta && "reasoning_content" in delta && delta.reasoning_content) {
				yield {
					type: "reasoning",
					// @ts-ignore-next-line
					reasoning: delta.reasoning_content,
				}
			}
		}
	}

	getModel(): { id: HuaweiCloudMaasModelId; info: ModelInfo } {
		const modelId = this.options.apiModelId
		if (modelId && modelId in huaweiCloudMaasModels) {
			const id = modelId as HuaweiCloudMaasModelId
			return { id, info: huaweiCloudMaasModels[id] }
		}
		return {
			id: huaweiCloudMaasDefaultModelId,
			info: huaweiCloudMaasModels[huaweiCloudMaasDefaultModelId],
		}
	}
}
