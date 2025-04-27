"server-only";

export interface PreprocessResponse {
    success: boolean;
}

export interface PreprocessOptions {
    category: "Upper-body" | "Lower-body" | "Dresses";
    offset_top?: number;
    offset_bottom?: number;
    offset_left?: number;
    offset_right?: number;
}

export async function takeCalibrationImage(
    image: File,
    options: PreprocessOptions
): Promise<PreprocessResponse> {
    const formData = new FormData();
    formData.append("vton_img", image);
    formData.append("category", options.category);

    try {
        const response = await fetch("/api/preprocess", {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        return { success: true };
    } catch (error) {
        // throw new Error(
        //     `Failed to preprocess image: ${error instanceof Error ? error.message : "Unknown error"}`
        // );
        return { success: false };
    }
}
