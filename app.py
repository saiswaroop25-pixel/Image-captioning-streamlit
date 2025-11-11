import streamlit as st
import requests
from PIL import Image
from io import BytesIO
from transformers import GPT2TokenizerFast, ViTImageProcessor, VisionEncoderDecoderModel
from gtts import gTTS


@st.cache_resource(show_spinner=False)
def load_models():
	model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
	tokenizer = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
	image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
	return model, tokenizer, image_processor


_WARMED_UP = False
def warmup_once():
	global _WARMED_UP
	if _WARMED_UP:
		return
	try:
		# Create a tiny blank image for a quick dry-run to load weights into memory
		img = Image.new("RGB", (64, 64), color=(128, 128, 128))
		model, tokenizer, image_processor = load_models()
		pixel_values = image_processor(img, return_tensors="pt").pixel_values
		_ = model.generate(pixel_values, max_new_tokens=8)
		_ = tokenizer.batch_decode(_, skip_special_tokens=True)[0]
	finally:
		_WARMED_UP = True


def generate_caption_from_image(pil_image: Image.Image) -> str:
	try:
		model, tokenizer, image_processor = load_models()
		pixel_values = image_processor(pil_image, return_tensors="pt").pixel_values
		generated_ids = model.generate(pixel_values, max_new_tokens=32)
		generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
		return generated_text.strip()
	except Exception as e:
		return f"Error generating caption: {str(e)}"


def read_image_from_url(image_url: str):
	resp = requests.get(image_url, stream=True, timeout=15)
	resp.raise_for_status()
	return Image.open(resp.raw).convert("RGB")


def main():
	st.set_page_config(page_title="Image Captioning + TTS", page_icon="🖼️", layout="centered")
	# Warm up model on first app run to reduce first-request latency
	warmup_once()
	st.title("🖼️ Image Captioning with Text-to-Speech")
	st.caption("Use a local image upload or paste an image URL.")

	with st.expander("Model info", expanded=False):
		st.write("Using `nlpconnect/vit-gpt2-image-captioning` from Hugging Face.")

	source = st.radio("Choose input source:", ["Upload image", "Image URL"], horizontal=True)

	pil_image = None
	if source == "Upload image":
		uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"])
		if uploaded is not None:
			try:
				pil_image = Image.open(uploaded).convert("RGB")
			except Exception as e:
				st.error(f"Could not read the image: {str(e)}")
	else:
		image_url = st.text_input("Enter Image URL")
		if image_url:
			try:
				pil_image = read_image_from_url(image_url)
			except Exception as e:
				st.error(f"Could not fetch image from URL: {str(e)}")

	if pil_image is not None:
		st.image(pil_image, caption="Selected Image", use_column_width=True)

	col1, col2 = st.columns([1, 1])
	with col1:
		generate = st.button("Generate Caption", type="primary", use_container_width=True)
	with col2:
		tts_btn = st.button("Listen to Caption Audio", use_container_width=True)

	if generate and pil_image is None:
		st.warning("Please provide an image (upload or URL).")

	# Hold caption in session for TTS
	if "caption_text" not in st.session_state:
		st.session_state.caption_text = ""

	if generate and pil_image is not None:
		with st.spinner("Generating caption..."):
			caption = generate_caption_from_image(pil_image)
		st.session_state.caption_text = caption
		st.subheader("Caption")
		st.write(st.session_state.caption_text or "No caption generated.")

	if tts_btn:
		if not st.session_state.get("caption_text"):
			st.warning("Generate a caption first.")
		else:
			try:
				tts = gTTS(st.session_state.caption_text, lang='en')
				audio_file = BytesIO()
				tts.write_to_fp(audio_file)
				audio_file.seek(0)
				st.audio(audio_file, format="audio/mp3")
			except Exception as e:
				st.error(f"Error generating audio: {str(e)}")


if __name__ == "__main__":
	main()