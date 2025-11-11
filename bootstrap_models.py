from transformers import GPT2TokenizerFast, ViTImageProcessor, VisionEncoderDecoderModel

def main():
	# Force download/cache of model and processors at build time
	VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
	GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
	ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

if __name__ == "__main__":
	main()


