import { cookies } from "next/headers";

export default async function Layout({
    children,
}: Readonly<{ children: React.ReactNode }>) {
    return (
        <div className="mx-auto flex aspect-[9/16] h-screen flex-col bg-white text-neutral-900">
            {children}
        </div>
    );
}
