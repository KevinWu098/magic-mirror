## Inspiration
More people than ever are shopping online. In the United States, **over 40% of all clothing** is purchased online. Shoppers love the convenience and wide array of options available online.

However, online consumers often report issues with sizing and fit, as well as products not looking how they had imagined when viewing it online. The increase in shoppers online, alongside this decrease in customer satisfaction, has resulted in rapidly increasing amounts of returned merchandise (30-40% returns, as of 2022).

Every year, roughly **40 billion dollars** of apparel is returned. Additionally, the cost to manufacture and also to return the goods can result in an additional 30% increase in greenhouse gas emissions. Shoppers are unhappy, sellers are losing money, and our environment is suffering because of rampant returns and low customer satisfaction.

That’s why we built Magic Mirror, the first interactive, virtual try-on studio enabling anyone to see, from the comfort of their own home, how clothes would look like on them. We aim to increase customer satisfaction, reduce clothing returns, and protect our planet using our innovative platform blending unique voice controls and our custom image-generation pipeline.

![image](https://github.com/user-attachments/assets/8397c8b4-5f01-4be7-baab-46b73a64f162)
## What it does

Magic Mirror uses a custom image generation pipeline to create highly realistic and accurate generations of how users would look when wearing certain pieces of clothing. This is powered by our pipeline which uses FastSAM and YOLO for segmentation. Then, we can either use a user provided image, or optionally use OpenAI’s image gen to create unique articles of clothing, which are placed onto users using FitDIT, an open-source model.

![image](https://github.com/user-attachments/assets/397f49e3-6586-4a4d-a4ab-e03247466863)
This system is accessed via our frontend, built with React and Next.js. The interface is controlled primarily via voice, with our conversational voice agent built on LiveKit. This allows for a completely hands-off and highly interactive experience.

## How we built it
Tech Stack:
- React
- Next.js
- TailwindCSS
- Python
- Pytorch
- H100 GPUs
- FastSAM
- Yolo
- LiveKit
- Flask
- FitDIT

