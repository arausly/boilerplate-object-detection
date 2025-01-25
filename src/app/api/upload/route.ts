import { HfInference } from "@huggingface/inference";

const hf = new HfInference(process.env.HF_TOKEN);

function blobToBase64(blob: Blob) {
    return new Promise((resolve, _) => {
        const reader = new FileReader();
        reader.onloadend = () => resolve(reader.result);
        reader.readAsDataURL(blob);
    });
}

export async function POST(request: Request) {
    const formData = await request.formData();

    const file = formData.get("file");
    const command = formData.get("command");

    if (!file || !command) {
        return Response.json(
            { error: true, msg: "Please provide file and command" },
            { status: 400 }
        );
    }

    try {
        const editedImageBlob = await hf.imageToImage({
            inputs: file,
            parameters: {
                prompt: command
            },
            model: "lllyasviel/sd-controlnet-depth"
        });

        return Response.json({
            error: false,
            msg: "successful",
            data: blobToBase64(editedImageBlob)
        });
    } catch (err) {
        console.log("Error ==>", err);
        return Response.json(
            { error: true, msg: "something went wrong" },
            {
                status: 500
            }
        );
    }
}
