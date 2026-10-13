"""The catalogue of what the studio may download and run.

Every file here is pinned by SHA-256 (the Hugging Face LFS object id) and size, and every
model carries its licence. Only permissive licences are listed: FLUX.1-dev, RMBG-2.0 and
other non-commercial weights are excluded by policy (ADR-051).
"""

from __future__ import annotations

from dataclasses import dataclass, field

GIB = 1024**3

HF = "https://huggingface.co/{repo}/resolve/main/{path}"


@dataclass(frozen=True)
class FileSpec:
    """One downloadable file."""

    filename: str
    url: str
    sha256: str
    size: int
    licence: str
    source: str  # human-readable origin


@dataclass(frozen=True)
class Variant:
    """A complete, runnable set of weights for one model, ranked by fidelity.

    ``rank`` 0 is the highest fidelity. The *best* preset always wants rank 0; a lower rank is
    used only when rank 0 does not fit in the machine's memory, and that is reported, never
    silent.
    """

    name: str
    rank: int
    files: tuple[FileSpec, ...]
    note: str

    @property
    def download_bytes(self) -> int:
        return sum(f.size for f in self.files)


@dataclass(frozen=True)
class Preset:
    """A quality/time trade-off chosen explicitly by the caller."""

    name: str
    width: int
    height: int
    steps: int
    cfg_scale: float
    tradeoff: str  # stated in words
    extra_args: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelSpec:
    name: str
    display_name: str
    licence: str
    family: str  # "flux" | "qwen-image"
    quality_rank: int  # 0 = strongest model in the catalogue
    variants: tuple[Variant, ...]
    presets: dict[str, Preset]
    sampler: str
    compute_factor: float  # relative cost per step per megapixel against FLUX.1-schnell
    description: str
    homepage: str
    components: dict[str, str] = field(default_factory=dict)  # sd.cpp flag -> filename prefix

    def variant(self, name: str) -> Variant:
        for v in self.variants:
            if v.name == name:
                return v
        raise KeyError(f"{self.name} has no variant {name!r}")

    @property
    def ranked_variants(self) -> list[Variant]:
        return sorted(self.variants, key=lambda v: v.rank)


# --- shared component files -------------------------------------------------------------

_CLIP_L = FileSpec(
    filename="clip_l.safetensors",
    url=HF.format(repo="comfyanonymous/flux_text_encoders", path="clip_l.safetensors"),
    sha256="660c6f5b1abae9dc498ac2d21e1347d2abdb0cf6c0c0c8576cd796491d9a6cdd",
    size=246144152,
    licence="Apache-2.0",
    source="comfyanonymous/flux_text_encoders",
)
_FLUX_AE = FileSpec(
    filename="flux-ae.safetensors",
    url=HF.format(repo="second-state/FLUX.1-schnell-GGUF", path="ae.safetensors"),
    sha256="afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38",
    size=335304388,
    licence="Apache-2.0",
    source="second-state/FLUX.1-schnell-GGUF (FLUX.1-schnell VAE, ungated mirror)",
)


def _t5(quant: str, sha: str, size: int) -> FileSpec:
    fn = f"t5-v1_1-xxl-encoder-{quant}.gguf"
    return FileSpec(
        filename=fn,
        url=HF.format(repo="city96/t5-v1_1-xxl-encoder-gguf", path=fn),
        sha256=sha,
        size=size,
        licence="Apache-2.0",
        source="city96/t5-v1_1-xxl-encoder-gguf",
    )


def _flux(quant: str, sha: str, size: int) -> FileSpec:
    fn = f"flux1-schnell-{quant}.gguf"
    return FileSpec(
        filename=fn,
        url=HF.format(repo="city96/FLUX.1-schnell-gguf", path=fn),
        sha256=sha,
        size=size,
        licence="Apache-2.0",
        source="city96/FLUX.1-schnell-gguf",
    )


_T5_Q8 = _t5("Q8_0", "9ec60f6028534b7fe5af439fcb535d75a68592a9ca3fcdeb175ef89e3ee99825", 5061584064)
_T5_Q5 = _t5(
    "Q5_K_M", "b51cbb10b1a7aac6dd1c3b62f0ed908bfd06e0b42d2f3577d43e061361f51dae", 3386856640
)
_T5_Q4 = _t5(
    "Q4_K_M", "6be2b0b7e2de7cf2919340c88cb802a103a997ce46c53131cec91958c1db1af4", 2896123072
)

FLUX_SCHNELL = ModelSpec(
    name="flux-schnell",
    display_name="FLUX.1 [schnell]",
    licence="Apache-2.0",
    family="flux",
    quality_rank=1,
    sampler="euler",
    compute_factor=1.0,
    homepage="https://huggingface.co/black-forest-labs/FLUX.1-schnell",
    description=(
        "Black Forest Labs' 12B rectified-flow transformer, step-distilled to 4 steps. "
        "Strong prompt following and detail; 4 steps is its full step count (more does not "
        "improve it). Text encoders: CLIP-L and T5-XXL."
    ),
    components={
        "--diffusion-model": "flux1-schnell",
        "--vae": "flux-ae",
        "--clip_l": "clip_l",
        "--t5xxl": "t5-v1_1-xxl",
    },
    variants=(
        Variant(
            "q8",
            0,
            (
                _flux(
                    "Q8_0",
                    "f6694941193b10148dbf1f0f498d4ccd3e9875c127fc53946213b68580c66f10",
                    12687821728,
                ),
                _T5_Q8,
                _CLIP_L,
                _FLUX_AE,
            ),
            "8-bit weights for the transformer and T5: visually indistinguishable from the "
            "16-bit original.",
        ),
        Variant(
            "q6",
            1,
            (
                _flux(
                    "Q6_K",
                    "a42fd143cec4d7194da281dc8d23a8fe54b16875a13423c042cb545d1da6fa50",
                    9834955808,
                ),
                _T5_Q8,
                _CLIP_L,
                _FLUX_AE,
            ),
            "6-bit transformer with 8-bit T5: very close to q8.",
        ),
        Variant(
            "q5",
            2,
            (
                _flux(
                    "Q5_K_S",
                    "19f63562d9a20d2123288dddd9abb3d5560554e5694ded951aaf6bfafe46b5df",
                    8263222304,
                ),
                _T5_Q5,
                _CLIP_L,
                _FLUX_AE,
            ),
            "5-bit transformer and T5: slight loss of fine texture.",
        ),
        Variant(
            "q4",
            3,
            (
                _flux(
                    "Q4_K_S",
                    "4fd16477b3a5296d0cf722c4b92a9fd7f30d09ac7495826e4465d8de9c9fd973",
                    6783943712,
                ),
                _T5_Q4,
                _CLIP_L,
                _FLUX_AE,
            ),
            "4-bit transformer and T5: the smallest set that still looks like FLUX; small "
            "text and fine detail degrade somewhat.",
        ),
    ),
    presets={
        "best": Preset(
            "best", 1024, 1024, 4, 1.0, "Full resolution (1024 px) and full steps. The default."
        ),
        "balanced": Preset(
            "balanced",
            768,
            768,
            4,
            1.0,
            "768 px instead of 1024: roughly half the time, less fine detail and a smaller image.",
        ),
        "fast": Preset(
            "fast",
            512,
            512,
            4,
            1.0,
            "512 px: roughly a quarter of the time, a small preview-grade image.",
        ),
    },
)


def _qwen(quant: str, sha: str, size: int) -> FileSpec:
    fn = f"qwen-image-{quant}.gguf"
    return FileSpec(
        filename=fn,
        url=HF.format(repo="city96/Qwen-Image-gguf", path=fn),
        sha256=sha,
        size=size,
        licence="Apache-2.0",
        source="city96/Qwen-Image-gguf",
    )


def _qwen_llm(quant: str, sha: str, size: int) -> FileSpec:
    fn = f"Qwen2.5-VL-7B-Instruct.{quant}.gguf"
    return FileSpec(
        filename=fn,
        url=HF.format(repo="mradermacher/Qwen2.5-VL-7B-Instruct-GGUF", path=fn),
        sha256=sha,
        size=size,
        licence="Apache-2.0",
        source="mradermacher/Qwen2.5-VL-7B-Instruct-GGUF (Qwen2.5-VL-7B-Instruct, Apache-2.0)",
    )


_QWEN_VAE = FileSpec(
    filename="qwen_image_vae.safetensors",
    url=HF.format(
        repo="Comfy-Org/Qwen-Image_ComfyUI", path="split_files/vae/qwen_image_vae.safetensors"
    ),
    sha256="a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f",
    size=253806246,
    licence="Apache-2.0",
    source="Comfy-Org/Qwen-Image_ComfyUI",
)
_QWEN_LLM_Q8 = _qwen_llm(
    "Q8_0", "577bfb3e41f93f00414e594c56a38bbe91207b3b636ab7faf343e34a20aa73ca", 8098524160
)
_QWEN_LLM_Q5 = _qwen_llm(
    "Q5_K_M", "993112f189a95ee82e01e993a3f1768c015d4fe70ec493ecf1b8f3f838a48b63", 5444830208
)
_QWEN_LLM_Q4 = _qwen_llm(
    "Q4_K_M", "0f00a930ba3108b6861ddadf74d8ebbd82e257c63eba728e62c3e8970f5eed94", 4683072512
)

QWEN_IMAGE = ModelSpec(
    name="qwen-image",
    display_name="Qwen-Image",
    licence="Apache-2.0",
    family="qwen-image",
    quality_rank=0,
    sampler="euler",
    compute_factor=3.4,  # 20B params vs 12B, and CFG runs two passes per step
    homepage="https://huggingface.co/Qwen/Qwen-Image",
    description=(
        "Alibaba's 20B image foundation model: the strongest permissively licensed generator "
        "in the catalogue, especially for rendered text and complex layouts. Needs a lot of "
        "memory (its text encoder alone is a 7B language model) and 50 steps with guidance."
    ),
    components={
        "--diffusion-model": "qwen-image",
        "--vae": "qwen_image_vae",
        "--llm": "Qwen2.5-VL",
    },
    variants=(
        Variant(
            "q8",
            0,
            (
                _qwen(
                    "Q8_0",
                    "43142b32778dc0568c27395abc6f8291ac53f1de0b98eed919712cf161f248de",
                    21761817120,
                ),
                _QWEN_LLM_Q8,
                _QWEN_VAE,
            ),
            "8-bit weights throughout.",
        ),
        Variant(
            "q6",
            1,
            (
                _qwen(
                    "Q6_K",
                    "d3331594de0eb342bad2b64bcfe9c9c12ae9cf4f3c6bda2851cc394d3ecc2593",
                    16824990240,
                ),
                _QWEN_LLM_Q8,
                _QWEN_VAE,
            ),
            "6-bit transformer, 8-bit text encoder.",
        ),
        Variant(
            "q5",
            2,
            (
                _qwen(
                    "Q5_K_S",
                    "55c5387542fcdad177b9df467c6d0b329727dcf197bfe28c819ed2face6fd474",
                    14117698080,
                ),
                _QWEN_LLM_Q5,
                _QWEN_VAE,
            ),
            "5-bit transformer and text encoder.",
        ),
        Variant(
            "q4",
            3,
            (
                _qwen(
                    "Q4_K_S",
                    "118ca698a0c0b77002a2ab00d4b1b7c79e2b4053210748740d0ca1fffc47fab0",
                    12140608032,
                ),
                _QWEN_LLM_Q4,
                _QWEN_VAE,
            ),
            "4-bit transformer and text encoder: the smallest usable set.",
        ),
    ),
    presets={
        "best": Preset(
            "best",
            1024,
            1024,
            50,
            2.5,
            "Full resolution, 50 steps with guidance 2.5. The default.",
            ("--flow-shift", "3"),
        ),
        "balanced": Preset(
            "balanced",
            1024,
            1024,
            30,
            2.5,
            "30 steps instead of 50: about 40% faster, slightly less refined.",
            ("--flow-shift", "3"),
        ),
        "fast": Preset(
            "fast",
            768,
            768,
            20,
            2.5,
            "768 px and 20 steps: roughly a quarter of the time, a preview-grade image.",
            ("--flow-shift", "3"),
        ),
    },
)

MODELS: dict[str, ModelSpec] = {m.name: m for m in (QWEN_IMAGE, FLUX_SCHNELL)}


# --- tool models: upscalers and background removal --------------------------------------


@dataclass(frozen=True)
class ToolModel:
    """A small single-file model behind a task tool (upscale, remove_background)."""

    name: str
    kind: str  # "upscaler" | "background"
    display_name: str
    licence: str
    file: FileSpec
    note: str
    default: bool = False
    scale: int = 1  # upscalers: output size factor


_GH = "https://github.com/{repo}/releases/download/{tag}/{name}"

TOOL_MODELS: dict[str, ToolModel] = {
    t.name: t
    for t in (
        ToolModel(
            "realesrgan-x4plus",
            "upscaler",
            "Real-ESRGAN x4plus",
            "BSD-3-Clause",
            FileSpec(
                "RealESRGAN_x4plus.pth",
                _GH.format(repo="xinntao/Real-ESRGAN", tag="v0.1.0", name="RealESRGAN_x4plus.pth"),
                "4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1",
                67040989,
                "BSD-3-Clause",
                "xinntao/Real-ESRGAN v0.1.0",
            ),
            "General-purpose 4x upscaler for photos and renders; the default.",
            default=True,
            scale=4,
        ),
        ToolModel(
            "realesrgan-anime-6b",
            "upscaler",
            "Real-ESRGAN x4plus anime 6B",
            "BSD-3-Clause",
            FileSpec(
                "RealESRGAN_x4plus_anime_6B.pth",
                _GH.format(
                    repo="xinntao/Real-ESRGAN",
                    tag="v0.2.2.4",
                    name="RealESRGAN_x4plus_anime_6B.pth",
                ),
                "f872d837d3c90ed2e05227bed711af5671a6fd1c9f7d7e91c911a61f155e99da",
                17938799,
                "BSD-3-Clause",
                "xinntao/Real-ESRGAN v0.2.2.4",
            ),
            "Smaller 4x upscaler tuned for flat colours and line art (anime, illustration).",
            scale=4,
        ),
        ToolModel(
            "isnet-general-use",
            "background",
            "IS-Net general use",
            "Apache-2.0",
            FileSpec(
                "isnet-general-use.onnx",
                _GH.format(repo="danielgatis/rembg", tag="v0.0.0", name="isnet-general-use.onnx"),
                "60920e99c45464f2ba57bee2ad08c919a52bbf852739e96947fbb4358c0d964a",
                178648008,
                "Apache-2.0",
                "danielgatis/rembg releases (IS-Net, Apache-2.0)",
            ),
            "Dichotomous image segmentation at 1024 px: clean edges on hair and thin structures.",
            default=True,
        ),
        ToolModel(
            "u2net",
            "background",
            "U²-Net",
            "Apache-2.0",
            FileSpec(
                "u2net.onnx",
                _GH.format(repo="danielgatis/rembg", tag="v0.0.0", name="u2net.onnx"),
                "8d10d2f3bb75ae3b6d527c77944fc5e7dcd94b29809d47a739a7a728a912b491",
                175997641,
                "Apache-2.0",
                "danielgatis/rembg releases (U²-Net, Apache-2.0)",
            ),
            "Classic salient-object model at 320 px: faster, softer edges.",
        ),
    )
}


# --- vision models for describe (image to text) ----------------------------------------


@dataclass(frozen=True)
class VisionModel:
    """A llama.cpp vision-language model: text weights plus multimodal projector."""

    name: str
    display_name: str
    licence: str
    model: FileSpec
    mmproj: FileSpec
    note: str
    default: bool = False

    @property
    def files(self) -> tuple[FileSpec, ...]:
        return (self.model, self.mmproj)

    @property
    def download_bytes(self) -> int:
        return self.model.size + self.mmproj.size


VISION_MODELS: dict[str, VisionModel] = {
    v.name: v
    for v in (
        VisionModel(
            "qwen2.5-vl-3b",
            "Qwen2.5-VL 3B Instruct",
            "Apache-2.0",
            FileSpec(
                "Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf",
                HF.format(
                    repo="ggml-org/Qwen2.5-VL-3B-Instruct-GGUF",
                    path="Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf",
                ),
                "d02fe9b69ad8cadbbd228e387667af66612c44bed29ffc8eb1e7caf9ac486c12",
                1929901056,
                "Apache-2.0",
                "ggml-org/Qwen2.5-VL-3B-Instruct-GGUF",
            ),
            FileSpec(
                "mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf",
                HF.format(
                    repo="ggml-org/Qwen2.5-VL-3B-Instruct-GGUF",
                    path="mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf",
                ),
                "980c9b2f78c04e6cff93d277ada09e768394f112d75db3b4e9dea8a69f9fb904",
                844757728,
                "Apache-2.0",
                "ggml-org/Qwen2.5-VL-3B-Instruct-GGUF",
            ),
            "Small vision-language model (2.8 GB): good captions and prompts, seconds on a CPU.",
            default=True,
        ),
        VisionModel(
            "qwen2.5-vl-7b",
            "Qwen2.5-VL 7B Instruct",
            "Apache-2.0",
            FileSpec(
                "Qwen2.5-VL-7B-Instruct.Q4_K_M.gguf",
                HF.format(
                    repo="mradermacher/Qwen2.5-VL-7B-Instruct-GGUF",
                    path="Qwen2.5-VL-7B-Instruct.Q4_K_M.gguf",
                ),
                "0f00a930ba3108b6861ddadf74d8ebbd82e257c63eba728e62c3e8970f5eed94",
                4683072512,
                "Apache-2.0",
                "mradermacher/Qwen2.5-VL-7B-Instruct-GGUF",
            ),
            FileSpec(
                "Qwen2.5-VL-7B-Instruct.mmproj-Q8_0.gguf",
                HF.format(
                    repo="mradermacher/Qwen2.5-VL-7B-Instruct-GGUF",
                    path="Qwen2.5-VL-7B-Instruct.mmproj-Q8_0.gguf",
                ),
                "2185ca0adf339f96845318c7b666f5d242f86bbace5657a89cb834785d0cea57",
                853119712,
                "Apache-2.0",
                "mradermacher/Qwen2.5-VL-7B-Instruct-GGUF",
            ),
            "Larger vision-language model (5.5 GB): richer descriptions, slower, more memory.",
        ),
    )
}


def default_vision_model() -> VisionModel:
    return next(v for v in VISION_MODELS.values() if v.default)


def default_tool_model(kind: str) -> ToolModel:
    return next(t for t in TOOL_MODELS.values() if t.kind == kind and t.default)


EXCLUDED_BY_POLICY: tuple[tuple[str, str], ...] = (
    ("FLUX.1 [dev]", "FLUX.1-dev Non-Commercial License"),
    ("RMBG-2.0", "non-commercial (CC BY-NC 4.0)"),
    ("Stable Diffusion 3.x / 3.5", "Stability AI Community License (revenue-gated)"),
)

PRESET_NAMES = ("best", "balanced", "fast")


# --- engine binaries ---------------------------------------------------------------------

SDCPP_TAG = "master-956-1b0ba10"
SDCPP_COMMIT = "1b0ba10"
_SDCPP_URL = "https://github.com/leejet/stable-diffusion.cpp/releases/download/{tag}/{asset}"


@dataclass(frozen=True)
class EngineAsset:
    platform: str  # sys.platform value
    arch: str  # normalised: x86_64 | arm64
    flavor: str  # cpu | vulkan
    asset: str
    sha256: str
    size: int
    exe: str  # name of the CLI inside the zip

    @property
    def url(self) -> str:
        if self.asset.startswith("llama-"):
            return _LLAMA_URL.format(tag=LLAMA_TAG, asset=self.asset)
        return _SDCPP_URL.format(tag=SDCPP_TAG, asset=self.asset)


SDCPP_ASSETS: tuple[EngineAsset, ...] = (
    EngineAsset(
        "win32",
        "x86_64",
        "cpu",
        f"sd-master-{SDCPP_COMMIT}-bin-win-cpu-x64.zip",
        "39c2ea1cc03f697864c858ad0b22b9b8ce62f945ba428e8596a814527e6db6fb",
        17543461,
        "sd-cli.exe",
    ),
    EngineAsset(
        "win32",
        "x86_64",
        "vulkan",
        f"sd-master-{SDCPP_COMMIT}-bin-win-vulkan-x64.zip",
        "9e8b9baff514fe5dfb77e51bf08d85f11e6f10128eac14d4474c87f563309725",
        30123908,
        "sd-cli.exe",
    ),
    EngineAsset(
        "linux",
        "x86_64",
        "cpu",
        f"sd-master-{SDCPP_COMMIT}-bin-Linux-Ubuntu-24.04-x86_64.zip",
        "4f6b7e110b641c311f7537b143aae33a3e17b532ead690e391deff972b567da2",
        26130308,
        "sd-cli",
    ),
    EngineAsset(
        "linux",
        "x86_64",
        "vulkan",
        f"sd-master-{SDCPP_COMMIT}-bin-Linux-Ubuntu-24.04-x86_64-vulkan.zip",
        "cecacb8054b4fab56912ee7e109a5f48311cf7a8660c6b311ac3a6eb1156ef73",
        36995435,
        "sd-cli",
    ),
    EngineAsset(
        "darwin",
        "arm64",
        "cpu",
        f"sd-master-{SDCPP_COMMIT}-bin-Darwin-macOS-26.6.2-arm64.zip",
        "72c18ca1ba4177ca77cd288a9cb1e0f78d63ad3bde4b91fd7c84f3f4187b5761",
        35158090,
        "sd-cli",
    ),
)


def normalise_arch(machine: str) -> str:
    m = machine.lower()
    if m in ("x86_64", "amd64"):
        return "x86_64"
    if m in ("arm64", "aarch64"):
        return "arm64"
    return m


def find_engine_asset(platform: str, machine: str, flavor: str) -> EngineAsset | None:
    arch = normalise_arch(machine)
    for a in SDCPP_ASSETS:
        if a.platform == platform and a.arch == arch and a.flavor == flavor:
            return a
    return None


LLAMA_TAG = "b11577"
_LLAMA_URL = "https://github.com/ggml-org/llama.cpp/releases/download/{tag}/{asset}"

LLAMA_ASSETS: tuple[EngineAsset, ...] = (
    EngineAsset(
        "win32",
        "x86_64",
        "cpu",
        "llama-b11577-bin-win-cpu-x64.zip",
        "c48e5762bb466a2faba1b880ca8b328a152e7f48e738431eb918ac9078f82ef1",
        19509426,
        "llama-mtmd-cli.exe",
    ),
    EngineAsset(
        "linux",
        "x86_64",
        "cpu",
        "llama-b11577-bin-ubuntu-x64.tar.gz",
        "619d152d8b7cb9ec5b738cec34b2620404c421c16f1ea51fc7a6d8a2b30771b6",
        17723303,
        "llama-mtmd-cli",
    ),
    EngineAsset(
        "darwin",
        "arm64",
        "cpu",
        "llama-b11577-bin-macos-arm64.tar.gz",
        "fa6526afe087293ac83709fe6529fa087e7070f606e30dd452173c16ccebed8c",
        12112274,
        "llama-mtmd-cli",
    ),
)


def find_llama_asset(platform: str, machine: str) -> EngineAsset | None:
    arch = normalise_arch(machine)
    for a in LLAMA_ASSETS:
        if a.platform == platform and a.arch == arch:
            return a
    return None


LICENCES: tuple[dict[str, str], ...] = (
    {"component": "Hivemind Image Studio", "licence": "Apache-2.0"},
    {"component": "stable-diffusion.cpp (engine)", "licence": "MIT"},
    {"component": "ggml", "licence": "MIT"},
    {"component": "FLUX.1 [schnell] weights (GGUF, city96)", "licence": "Apache-2.0"},
    {"component": "T5-XXL v1.1 encoder (GGUF, city96)", "licence": "Apache-2.0"},
    {"component": "CLIP-L text encoder (comfyanonymous)", "licence": "Apache-2.0"},
    {"component": "FLUX VAE (ae.safetensors)", "licence": "Apache-2.0"},
    {"component": "Qwen-Image weights (GGUF, city96)", "licence": "Apache-2.0"},
    {"component": "Qwen2.5-VL-7B-Instruct text encoder (GGUF)", "licence": "Apache-2.0"},
    {"component": "Qwen-Image VAE (Comfy-Org repack)", "licence": "Apache-2.0"},
    {"component": "Real-ESRGAN upscalers (xinntao)", "licence": "BSD-3-Clause"},
    {"component": "IS-Net and U²-Net background models", "licence": "Apache-2.0"},
    {"component": "ONNX Runtime", "licence": "MIT"},
    {"component": "llama.cpp (vision engine)", "licence": "MIT"},
    {"component": "Qwen2.5-VL Instruct (GGUF)", "licence": "Apache-2.0"},
)
