"use client";
import React from "react";

import { useDropzone } from "react-dropzone";
import {
    pipeline,
    ObjectDetectionPipelineOutput
} from "@huggingface/transformers";
import { Spinner } from "@/components/ui/spinner";
import { Button } from "@/components/ui/button";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { Badge } from "@/components/ui/badge";

enum STATUS_STATES {
    idle = "idle",
    loading = "Loading",
    ready = "ready",
    done = "done",
    detecting = "detecting",
    finished = "finished",
    failed = "Couldn't detect"
}

const getRandomColor = () => {
    const colors = [
        "#B82132",
        "#EFB036",
        "#818C78",
        "#23486A",
        "#872341",
        "#3D3D3D",
        "#F0BB78",
        "#441752",
        "#1F4529",
        "#3B1C32",
        "#3B3030"
    ];
    return colors[Math.floor(Math.random() * colors.length - 1)];
};

export default function ImageDetectionPage() {
    const [imageSRC, setImageSRC] = React.useState<string>("");
    const [transformer, setTransformer] = React.useState<any>(null);
    const [reload, setReload] = React.useState<number>(0);
    const [status, setStatus] = React.useState<STATUS_STATES>(
        STATUS_STATES.idle
    );
    const [matchBoxes, setMatchBoxes] = React.useState<any[]>([]);
    const [foundObjects, setFoundObjects] = React.useState<
        ObjectDetectionPipelineOutput
    >([]);
    const [selectedLabels, setSelectedLabels] = React.useState<
        Map<string, boolean>
    >(new Map());
    const [labelColorConfig, setLabelColorConfig] = React.useState<
        Map<string, string>
    >(new Map());

    //load transformer
    React.useEffect(() => {
        (async () => {
            try {
                //another model Xenova/detr-resnet-50
                //Xenova/yolos-tiny
                setStatus(STATUS_STATES.loading);
                const transformer = await pipeline(
                    "object-detection",
                    "Xenova/detr-resnet-50"
                );
                if (transformer) {
                    setTransformer(transformer);
                }
            } catch (err) {
                console.error("ERROR ==>", err);
            } finally {
                setStatus(STATUS_STATES.ready);
            }
        })();
    }, [reload]);

    const onDrop = React.useCallback((files: File[]) => {
        const file = files[0];
        setImageSRC(URL.createObjectURL(file));
    }, []);

    const { getRootProps, getInputProps } = useDropzone({
        onDrop,
        maxFiles: 1
    });

    const showSpinner = React.useMemo(() => {
        switch (status) {
            case STATUS_STATES.loading:
            case STATUS_STATES.detecting:
                return true;
            default:
                return false;
        }
    }, [status]);

    const detectObjects = React.useCallback(async () => {
        if (!imageSRC || !transformer) return;
        try {
            setStatus(STATUS_STATES.detecting);
            const detector = await pipeline(
                "object-detection",
                "Xenova/yolos-tiny"
            );

            const detectedObjects = (await detector(imageSRC, {
                threshold: 0.7,
                percentage: true
            })) as ObjectDetectionPipelineOutput;
            if (!detectedObjects.length) throw new Error();
            setFoundObjects(detectedObjects);
            //select all labels
            setSelectedLabels(
                () => new Map(detectedObjects.map((d) => [d.label, true]))
            );
            setStatus(STATUS_STATES.finished);
        } catch (err) {
            console.error("Error detecting ==>", err);
            setStatus(STATUS_STATES.failed);
        }
    }, [imageSRC, transformer]);

    const createMatchBoxes = React.useCallback(
        (foundObjects: ObjectDetectionPipelineOutput) => {
            if (!foundObjects.length) return;
            setMatchBoxes(() =>
                foundObjects.map((foundObject) => {
                    const { label } = foundObject;
                    const color =
                        labelColorConfig.get(label) ?? getRandomColor();
                    if (!labelColorConfig.has(label)) {
                        setLabelColorConfig(
                            (prev) => new Map([...prev, [label, color]])
                        );
                    }
                    const { xmin, xmax, ymax, ymin } = foundObject.box;
                    return (
                        <div
                            style={{
                                position: "absolute",
                                border: 2,
                                borderStyle: "solid",
                                borderColor: color,
                                top: `${ymin * 100}%`,
                                left: `${xmin * 100}%`,
                                width: `${100 * (xmax - xmin)}%`,
                                height: `${100 * (ymax - ymin)}%`
                            }}
                        >
                            <div
                                style={{
                                    backgroundColor: color,
                                    left: 0,
                                    position: "absolute",
                                    bottom: 0,
                                    transform: "translateX(-100%)"
                                }}
                            >
                                <p className="text-white text-nowrap text-center">
                                    {foundObject.label}
                                </p>
                            </div>
                        </div>
                    );
                })
            );
        },
        [labelColorConfig]
    );

    const foundLabels = Object.entries(
        foundObjects.reduce((hashMap, entry) => {
            if (!hashMap[entry.label]) {
                hashMap[entry.label] = 1;
            } else {
                ++hashMap[entry.label];
            }
            return hashMap;
        }, {} as Record<string, number>)
    );

    const handleToggleSelection = React.useCallback(
        (label: string) => {
            setSelectedLabels(
                (prevLabels) =>
                    new Map([...prevLabels, [label, !prevLabels.get(label)]])
            );
        },
        [selectedLabels]
    );

    React.useEffect(() => {
        createMatchBoxes(
            foundObjects.filter((o) => selectedLabels.get(o.label))
        );
    }, [selectedLabels, foundObjects]);

    return (
        <div className="w-screen h-screen flex justify-center font-[family-name:var(--font-geist-sans)]">
            <section className="flex flex-col w-2/3 h-full p-8">
                <div className="flex items-center justify-between mb-4">
                    <h2 className="text-2xl font-semibold">
                        Detect objects in image
                    </h2>
                    <span className="flex items-center space-x-2">
                        <p>Status: </p>
                        <p className="capitalize">{status}</p>
                        {showSpinner && <Spinner size="tiny" />}
                    </span>
                </div>
                <div
                    className="relative w-full h-3/4 border-t border rounded-lg mb-4 flex justify-center items-center cursor-pointer"
                    {...getRootProps()}
                >
                    <input {...getInputProps()} />

                    {imageSRC ? (
                        <div className="h-full w-full rounded-md overflow-hidden">
                            <img
                                src={imageSRC}
                                alt="Uploaded Preview"
                                className="h-full w-full object-cover"
                            />
                        </div>
                    ) : (
                        <p>
                            Drag 'n' drop some files here, or click to select
                            files
                        </p>
                    )}
                    {matchBoxes}
                </div>
                <div className="flex items-center justify-end space-x-8">
                    <Button
                        variant="secondary"
                        onClick={() => setReload((p) => ++p)}
                    >
                        Reload transformer
                    </Button>
                    <Button
                        disabled={!imageSRC || showSpinner}
                        onClick={detectObjects}
                    >
                        Detect objects
                    </Button>
                </div>
            </section>
            <aside className="pt-20">
                <ToggleGroup
                    type="single"
                    className="flex flex-col space-y-4 justify-start items-start"
                >
                    {foundLabels.map(([label, count]) => (
                        <ToggleGroupItem
                            key={label}
                            value={label}
                            className="space-x-2"
                            defaultChecked
                            onClick={() => handleToggleSelection(label)}
                        >
                            <span
                                className="inline-block w-5 h-5 rounded-md"
                                style={{
                                    backgroundColor: labelColorConfig.get(label)
                                }}
                            />
                            <p>{label}</p>
                            <Badge>{count}</Badge>
                        </ToggleGroupItem>
                    ))}
                </ToggleGroup>
            </aside>
        </div>
    );
}
