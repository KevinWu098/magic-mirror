"server-only";

export async function findSimilarClothing(garment: File) {
    const garmentBase64 = await new Promise<string>((resolve) => {
        const reader = new FileReader();
        reader.onloadend = () => {
            const base64String = reader.result as string;
            resolve(base64String.split(",").at(1) ?? "");
        };
        reader.readAsDataURL(garment);
    });

    const response = await fetch(
        "https://jos9w7i1fq624g-5000.proxy.runpod.net/search-lens",
        {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                image_base64: garmentBase64,
            }),
        }
    );

    return response.json();
}
