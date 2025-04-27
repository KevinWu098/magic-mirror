"server-only";

export async function takeCalibrationImage(
    image: File
): Promise<{ status: string }> {
    // Convert File to base64
    const base64 = await new Promise<string>((resolve) => {
        const reader = new FileReader();
        reader.onloadend = () => {
            const base64String = reader.result as string;
            // Remove the data URL prefix (e.g., "data:image/jpeg;base64,")
            resolve(base64String.split(",").at(1) ?? "");
        };
        reader.readAsDataURL(image);
    });

    try {
        fetch("https://jos9w7i1fq624g-5000.proxy.runpod.net/preprocess", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                vton_img_base64: base64,
            }),
        });

        return { status: "dispatched" };
    } catch (error) {
        return { status: "preprocessing_completed_with_errors" };
    }
}

export async function generateTryOn(
    garment: File,
    category: "Lower-body" | "Upper-body" | "Dresses"
): Promise<{ result: string }> {
    // Convert File to base64
    const base64 = await new Promise<string>((resolve) => {
        const reader = new FileReader();
        reader.onloadend = () => {
            const base64String = reader.result as string;
            // Remove the data URL prefix (e.g., "data:image/jpeg;base64,")
            resolve(base64String.split(",").at(1) ?? "");
        };
        reader.readAsDataURL(garment);
    });

    try {
        const response = await fetch(
            "https://jos9w7i1fq624g-5000.proxy.runpod.net/infer",
            {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    garm_img_base64: base64,
                    category: category,
                }),
            }
        );

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        return { result: data.result_image_base64 };
    } catch (error) {
        console.error("Error during inference:", error);
        throw error;
    }
}

export async function generateClothing(
    garment: File,
    category: "Lower-body" | "Upper-body" | "Dresses",
    vtonImage: File
): Promise<{ result: string }> {
    // Convert garment File to base64
    const garmentBase64 = await new Promise<string>((resolve) => {
        const reader = new FileReader();
        reader.onloadend = () => {
            const base64String = reader.result as string;
            resolve(base64String.split(",").at(1) ?? "");
        };
        reader.readAsDataURL(garment);
    });

    // Convert vton image File to base64
    const vtonBase64 = await new Promise<string>((resolve) => {
        const reader = new FileReader();
        reader.onloadend = () => {
            const base64String = reader.result as string;
            resolve(base64String.split(",").at(1) ?? "");
        };
        reader.readAsDataURL(vtonImage);
    });

    console.log(garmentBase64, vtonBase64);

    try {
        const response = await fetch(
            "https://jos9w7i1fq624g-5000.proxy.runpod.net/generate",
            {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    category: category,
                    garm_img_base64: garmentBase64,
                    vton_img_base64: vtonBase64,
                }),
            }
        );

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        return { result: data.result_image_base64 };
    } catch (error) {
        console.error("Error during generation:", error);
        throw error;
    }
}
