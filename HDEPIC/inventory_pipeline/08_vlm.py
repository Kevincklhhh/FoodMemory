# Qwen3-VL API endpoint
QWEN3VL_URL = "http://saltyfish.eecs.umich.edu:8000/v1/chat/completions"
QWEN_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"


class VLMClient:
    """Handles communication with VLM APIs (Qwen and GPT-4o)"""

    def __init__(self, model_name: str = 'qwen', use_video: bool = True):
        self.model_name = model_name
        self.use_video = use_video
        self.openai_api = None

        if model_name == 'gpt-4o':
            if OpenAIAPI is None:
                raise ImportError("OpenAIAPI not found. Cannot use gpt-4o.")
            print(f"[VLMClient] Initializing GPT-4o API...")
            self.openai_api = OpenAIAPI(deployment='gpt-4o')
            # GPT-4o in this pipeline is text-only for now as per requirements
            self.use_video = False 

    def encode_video_base64(self, video_path: Path) -> str:
        """Encode video file to base64"""
        with open(video_path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    def query(
        self,
        system_prompt: str,
        user_prompt: str,
        video_path: Optional[Path] = None,
        max_tokens: int = 2000,
        temperature: float = 0.3
    ) -> str:
        """Dispatch query to appropriate model"""
        if self.model_name == 'gpt-4o':
            return self._query_openai(system_prompt, user_prompt, max_tokens)
        else:
            return self._query_qwen(system_prompt, user_prompt, video_path, max_tokens, temperature)

    def _query_openai(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        """Query GPT-4o (Text Only)"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        try:
            completion = self.openai_api.chat_completion(messages, max_tokens=max_tokens)
            return completion.choices[0].message.content
        except Exception as e:
            print(f"  ✗ OpenAI API Error: {e}")
            return ""

    def _query_qwen(
        self,
        system_prompt: str,
        user_prompt: str,
        video_path: Optional[Path],
        max_tokens: int,
        temperature: float
    ) -> str:
        """Query Qwen3-VL"""
        messages = [{"role": "system", "content": system_prompt}]
        
        user_content = []
        if self.use_video and video_path and video_path.exists():
            video_base64 = self.encode_video_base64(video_path)
            user_content.append({
                "type": "video_url",
                "video_url": {"url": f"data:video/mp4;base64,{video_base64}"}
            })
        
        user_content.append({"type": "text", "text": user_prompt})
        messages.append({"role": "user", "content": user_content})

        data = {
            "model": QWEN_MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        
        if self.use_video and video_path and video_path.exists():
            data["extra_body"] = {
                "mm_processor_kwargs": {
                    "fps": 1,
                    "do_sample_frames": True
                }
            }

        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(QWEN3VL_URL, headers=headers, json=data, timeout=180)
            response.raise_for_status()
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            return ""
        except Exception as e:
            print(f"  ✗ Qwen API Error: {e}")
            return ""