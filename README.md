# noScribe
### Cutting Edge AI Technology for Automated Audio Transcription
</br>

> [!IMPORTANT]
> ### 🍎 This is a fork: noScribe with the Voxtral engine for Apple Silicon
> On top of everything noScribe does, this fork adds a second transcription
> engine — Mistral's **Voxtral**, running locally through MLX on Apple Silicon.
> On hard conversational German it transcribes noticeably better than Whisper:
> **4.3 % against 8.1 % word error** on a hand-corrected interview passage, with
> denser punctuation, and it omits where Whisper invents. Everything else about
> noScribe is unchanged, and it still runs entirely on your own machine.
>
> **[→ What it is, how to install it, and every measurement behind it: VOXTRAL.md](VOXTRAL.md)**
>
> Requires a Mac with Apple Silicon (M1–M4). There is **no packaged download for
> this fork yet** — you install it from source, on top of a normal noScribe
> checkout. Everything else in this README is the original project's and applies
> unchanged.
>
> **Looking for the official noScribe?** It lives at
> [kaixxx/noScribe](https://github.com/kaixxx/noScribe) and
> [noscribe.de](https://noscribe.de) — that is where the ready-made downloads for
> Windows, macOS and Linux are, and where to go for support. This fork exists to
> try the Voxtral engine out in public; the fixes in it are offered back upstream
> as pull requests.

---

> [!NOTE]
> ### 🚀 The new official website for noScribe: https://noscribe.de
> Learn how to install and use the software, and find tips to improve transcription quality.
>
> 🌐 Available in **English, German, Spanish, Italian, and Dutch**.
>
> Please update your links. 

---

> [!WARNING]
> Somebody has registered the domain **noscribe(dot)ai** to sell transcription services. **Stay away from this platform, I have nothing to do with it.** The real noScribe is free and always will be. This is obviously an attempt to profit from the popularity of my software and the reputation it gained over the years. Very sad. 

## What is noScribe?
- An app to produce **high quality transcripts of interviews** for qualitative social research or journalistic use
- noScribe is **free and open source** ([GPL-3.0](https://www.gnu.org/licenses/gpl-3.0.html)), available for Windows, MacOS and Linux 
- It runs **completely locally** on your computer, protecting the confidentiality of your interviews. No cloud, no worries
- It can distinguish between different **speakers** and understands around 60 languages (more or less, see below)
- It includes a **nice editor** to review, verify and correct the resulting transcript
- It is standing on the shoulders of giants: [Whisper from OpenAI](https://github.com/openai/whisper), [faster-whisper by Guillaume Klein](https://github.com/guillaumekln/faster-whisper) and [pyannote from Hervé Bredin](https://github.com/pyannote/pyannote-audio)

</br>

![Main window](img/noScribe_main_window.png)
(The transcript is from [this interview](https://www.youtube.com/watch?v=vOwajAbvPzQ&t=2018s) which I did in May 2022 with the Russian sociologist Natalia Savelyeva.)

## Download, Installation, and Usage

See the corresponding sections on my website: https://noscribe.de 

## About Me
**Kai Dröge**, PhD in sociology (with a background in computer science), qualitative researcher and teacher, [Lucerne University for Applied Science (Switzerland)](https://www.hslu.ch/de-ch/hochschule-luzern/ueber-uns/personensuche/profile/?pid=823) and [Institute for Social Research, Frankfurt/M. (Germany)](https://www.ifs.uni-frankfurt.de/personendetails/kai-droege.html).

## Donate
NoScribe is free and always will be. However, developing it costs real money. I have purchased hardware for testing and pay Apple annually for a developer ID. If you would like to support this project, you can make a donation on Ko-Fi. Thanks! 

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/noscribe)

## Citation (APA Style)
Dröge, K. (2026). noScribe. AI-powered Audio Transcription (Version XXX) [Computer software]. https://github.com/kaixxx/noScribe

## Other Software
If you are interested in open source software for the analysis of qualitative data, take a look at [QualCoder](https://github.com/ccbogel/QualCoder), where I am a regular contributor.





