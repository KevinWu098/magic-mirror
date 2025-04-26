import { Client } from "@/components/client";
import { generateUUID } from "@/lib/utils";

export default function Page() {
    const id = generateUUID();

    return <Client id={id} />;
}
