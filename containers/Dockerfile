FROM vllm/vllm-openai:v0.29.0@sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1 AS build
ARG SOURCE_REVISION
ARG SOURCE_DATE_EPOCH=315532800
ENV SOURCE_REVISION=${SOURCE_REVISION} SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH}
ENV NVIDIA_VISIBLE_DEVICES=void TORCH_CUDA_ARCH_LIST="8.6;8.9;12.0" MAX_JOBS=2 PYTHONUTF8=1
WORKDIR /src
COPY build.py fp8_attn.cu prefill.cu pyproject.toml LICENSE NOTICE fa2_prefill.py paged_prefill.py ./
COPY headers ./headers
COPY licenses ./licenses
COPY integration/vllm_plugin ./integration/vllm_plugin
COPY containers/build_artifacts.py ./containers/build_artifacts.py
RUN python3 containers/build_artifacts.py --out /artifacts

FROM busybox:1.37.0@sha256:9db7b59979c38555a39def84a31fb98b5296952f9e3afd4f6f11f05b07adfab0
ARG SOURCE_REVISION
LABEL org.opencontainers.image.source="https://github.com/AntonProkopyev/fa2-fp8kv-sm86"
LABEL org.opencontainers.image.revision=${SOURCE_REVISION}
ENV NVIDIA_VISIBLE_DEVICES=void
COPY --from=build /artifacts /artifacts
COPY containers/export.sh /export.sh
ENTRYPOINT ["/bin/sh", "/export.sh"]
