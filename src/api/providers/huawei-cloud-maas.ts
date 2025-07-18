import { ApiHandler } from ".."
import { huaweiCloudMaaSDefaultModelId, HuaweiCloudMaaSModelId, huaweiCloudMaaSModels, ModelInfo } from "@shared/api"
import { Anthropic } from "@anthropic-ai/sdk"
import OpenAI from "openai"
import { convertToOpenAiMessages } from "../transform/openai-format"
import { ApiStream } from "../transform/stream"
import { withRetry } from "../retry"

interface HuaweiCloudMaaSHandlerOptions {
	huaweiCloudMaaSApiKey?: string
	apiModelId?: string
}

export class HuaweiCloudMaaSHandler implements ApiHandler {
	private options: HuaweiCloudMaaSHandlerOptions
	private client: OpenAI | undefined
	constructor(options: HuaweiCloudMaaSHandlerOptions) {
		this.options = options
	}

	private ensureClient(): OpenAI {
		if (!this.client) {
			if (!this.options.huaweiCloudMaaSApiKey) {
				throw new Error("Huawei Cloud MaaS API key is required")
			}
			try {
				this.client = new OpenAI({
					baseURL: "https://api.modelarts-maas.com/v1/",
					apiKey: this.options.huaweiCloudMaaSApiKey,
				})
			} catch (error) {
				throw new Error(`Error creating Huawei Cloud MaaS client: ${error.message}`)
			}
		}
		return this.client
	}

	getModel(): { id: HuaweiCloudMaaSModelId; info: ModelInfo } {
		const modelId = this.options.apiModelId
		if (modelId && modelId in huaweiCloudMaaSModels) {
			const id = modelId as HuaweiCloudMaaSModelId
			return { id, info: huaweiCloudMaaSModels[id] }
		}
		return {
			id: huaweiCloudMaaSDefaultModelId,
			info: huaweiCloudMaaSModels[huaweiCloudMaaSDefaultModelId],
		}
	}

	@withRetry()
	async *createMessage(systemPrompt: string, messages: Anthropic.Messages.MessageParam[]): ApiStream {
		const client = this.ensureClient()
		const model = this.getModel()
		let openAiMessages: OpenAI.Chat.ChatCompletionMessageParam[] = [
			{ role: "system", content: systemPrompt },
			...convertToOpenAiMessages(messages),
		]
		const stream = await client.chat.completions.create({
			model: model.id,
			max_completion_tokens: model.info.maxTokens,
			messages: openAiMessages,
			stream: true,
			stream_options: { include_usage: true },
			temperature: 0,
		})

		let reasoning: string | null = null
		let didOutputUsage: boolean = false
		let finalUsage: any = null

		for await (const chunk of stream) {
			const delta = chunk.choices[0]?.delta

			// Handle reasoning content detection
			if (delta?.content) {
				if (reasoning || delta.content.includes("<think>")) {
					reasoning = (reasoning || "") + delta.content
				} else if (!reasoning) {
					yield {
						type: "text",
						text: delta.content,
					}
				}
			}

			// Handle reasoning output
			if (reasoning || (delta && "reasoning_content" in delta && delta.reasoning_content)) {
				const reasoningContent = delta?.content || ((delta as any)?.reasoning_content as string | undefined) || ""
				if (reasoningContent.trim()) {
					yield {
						type: "reasoning",
						reasoning: reasoningContent,
					}
				}

				// Check if reasoning is complete
				if (reasoning?.includes("</think>")) {
					reasoning = null
				}
			}

			// Store usage information for later output
			if (chunk.usage) {
				finalUsage = chunk.usage
			}

			// Output usage when stream is finished
			if (!didOutputUsage && chunk.choices?.[0]?.finish_reason) {
				if (finalUsage) {
					yield {
						type: "usage",
						inputTokens: finalUsage.prompt_tokens || 0,
						outputTokens: finalUsage.completion_tokens || 0,
						cacheWriteTokens: 0,
						cacheReadTokens: 0,
					}
				}
				didOutputUsage = true
			}
		}
	}
}
