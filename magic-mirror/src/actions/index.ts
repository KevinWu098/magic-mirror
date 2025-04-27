"server-only";

export async function takeCalibrationImage(image: string) {
    try {
        await fetch("/api/take-calibration-image", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ image }),
        });

        return { success: "foo" };
    } catch {
        return { success: "bar" };
    }
}
