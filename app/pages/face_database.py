import os

from PIL import Image

import numpy as np

import pandas as pd
import streamlit as st
from qdrant_client import models

from face_voice_rec import config
from face_voice_rec.domain.face.face_core import FaceIndex
from face_voice_rec.storage.qdrant_store import get_qdrant_vector_store
from face_voice_rec.storage.database import (
    create_person,
    delete_low_quality_unknowns,
    delete_unknown,
    list_persons,
    list_unknown_samples,
    list_unknowns,
    resolve_unknown,
)


@st.cache_resource
def get_face_index():
    return FaceIndex()


def render() -> None:
    st.header("👤 Face Database")

    index = get_face_index()

    persons = list_persons()

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Known Persons", len(persons))

    with col2:
        face_embedding_count = get_qdrant_vector_store().count(
            config.QDRANT_FACE_COLLECTION,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="record_type",
                        match=models.MatchValue(value="enrolled_embedding"),
                    ),
                    models.FieldCondition(
                        key="modality",
                        match=models.MatchValue(value="face"),
                    ),
                ]
            ),
        )
        st.metric("Face Embeddings", face_embedding_count)

    if persons:
        st.subheader("Known Persons")

        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Person ID": p[0],
                        "Name": p[1],
                        "Created": p[2],
                    }
                    for p in persons
                ]
            ),
            width="stretch",
        )

    st.markdown("---")
    st.subheader("➕ Add a Person")

    with st.form("add_person"):
        name = st.text_input("Person name")
        submitted = st.form_submit_button("Create Person")

        if submitted:
            if not name.strip():
                st.error("Name cannot be empty.")
            else:
                create_person(name)
                st.success(f"Created person: {name}")
                st.rerun()

    st.markdown("---")
    st.subheader("❓ Pending Unknown People")

    unknowns = list_unknowns()

    if not unknowns:
        st.success("No unresolved unknown faces.")
        return

    st.info(
        "Review the captured samples below. "
        "You can assign an unknown face to an existing person "
        "or create a new person from the captured face."
    )

    if st.button(
        "🗑️ Delete Unknown Faces Below 40% Quality",
        key="delete_low_quality_unknowns",
        help="Permanently removes unresolved unknown faces whose best sample quality is below 0.40.",
    ):
        deleted_ids = delete_low_quality_unknowns(0.40)
        if deleted_ids:
            st.success(f"Deleted {len(deleted_ids)} low-quality unknown face(s).")
            get_face_index().rebuild()
            st.rerun()
        else:
            st.info("No unresolved unknown faces below 40% quality were found.")

    people = list_persons()
    options = {p[1]: p[0] for p in people}

    for (
        unknown_id,
        label,
        image_path,
        embedding_path,
        created,
        _resolved,
    ) in unknowns:
        samples = list_unknown_samples(unknown_id)

        sample_records = []

        for sample in samples:
            (
                sample_id,
                sample_image_path,
                sample_embedding_path,
                quality,
                angle_score,
                sample_created,
            ) = sample

            if (
                sample_image_path
                and os.path.exists(sample_image_path)
                and sample_embedding_path
                and os.path.exists(sample_embedding_path)
            ):
                sample_records.append({
                    "sample_id": sample_id,
                    "image_path": sample_image_path,
                    "embedding_path": sample_embedding_path,
                    "quality": float(quality),
                    "angle_score": float(angle_score),
                    "created": sample_created,
                })

        if not sample_records:
            if (
                image_path
                and str(image_path).lower() != "none"
                and os.path.exists(image_path)
                and embedding_path
                and os.path.exists(embedding_path)
            ):
                sample_records.append({
                    "sample_id": None,
                    "image_path": image_path,
                    "embedding_path": embedding_path,
                    "quality": 0.0,
                    "angle_score": 0.0,
                    "created": created,
                })

        sample_records.sort(
            key=lambda x: x["quality"],
            reverse=True,
        )

        sample_count = len(sample_records)

        best_image = (
            sample_records[0]["image_path"]
            if sample_records
            else None
        )

        best_quality = (
            sample_records[0]["quality"]
            if sample_records
            else 0.0
        )

        with st.container(border=True):
            st.markdown(
                f"### ❓ Unknown #{unknown_id}"
            )

            st.caption(
                f"Track: `{label}`"
            )

            st.caption(
                f"Captured: {sample_count} sample"
                f"{'s' if sample_count != 1 else ''}"
                f"  •  Created: {created}"
            )

            if st.button(
                "🗑️ Delete Unknown",
                key=f"delete_unknown_{unknown_id}",
                help="Permanently delete this unresolved unknown face and its saved samples.",
            ):
                if delete_unknown(unknown_id):
                    get_face_index().rebuild()
                    st.rerun()
                else:
                    st.warning("This unknown face is no longer unresolved.")

            if best_image and os.path.exists(best_image):
                st.image(
                    best_image,
                    caption=f"Best sample • Quality {best_quality:.2f}",
                    width=300,
                )

            else:
                st.warning(
                    "No valid face image is available for this "
                    "unknown record."
                )

            if sample_records:
                st.markdown(
                    f"**Face Samples ({sample_count})**"
                )

                columns = st.columns(
                    min(6, max(1, sample_count))
                )

                for i, sample in enumerate(sample_records):
                    with columns[i % len(columns)]:
                        if os.path.exists(sample["image_path"]):
                            try:
                                img = Image.open(sample["image_path"]).convert("RGB")

                                img.thumbnail((180, 180))

                                canvas = Image.new(
                                    "RGB",
                                    (180, 180),
                                    "white",
                                )

                                x = (180 - img.width) // 2
                                y = (180 - img.height) // 2

                                canvas.paste(img, (x, y))

                                st.image(
                                    canvas,
                                    width=180,
                                )

                                st.caption(
                                    f"#{i + 1} • Quality: "
                                    f"{sample['quality']:.2f}"
                                )

                            except Exception as exc:
                                st.warning(
                                    f"Could not display sample: {exc}"
                                )

            else:
                st.info(
                    "No individual samples were found. "
                    "Using the original unknown embedding."
                )

            st.markdown("---")

            if options:
                st.markdown("#### 👤 Assign to Existing Person")

                selected_name = st.selectbox(
                    "Select person",
                    ["-- Select --"] + list(options.keys()),
                    key=f"unknown_person_{unknown_id}",
                )

                if st.button(
                    "✅ Assign Existing Person",
                    key=f"assign_{unknown_id}",
                    type="primary",
                ):
                    if selected_name == "-- Select --":
                        st.warning(
                            "Please select a person first."
                        )

                    else:
                        selected_person_id = options[selected_name]

                        embeddings_added = 0

                        for sample in sample_records:
                            try:
                                emb = np.load(
                                    sample["embedding_path"],
                                    allow_pickle=True,
                                )
                                if emb.dtype == object:
                                    emb = np.asarray(emb.tolist(), dtype=np.float32)

                                if emb is None:
                                    continue

                                from face_voice_rec.domain.face.face_core import (
                                    save_embedding_for_person,
                                )

                                save_embedding_for_person(
                                    selected_name,
                                    emb,
                                    image_path=sample["image_path"],
                                    quality=sample["quality"],
                                )

                                embeddings_added += 1

                            except Exception as exc:
                                st.warning(
                                    f"Could not add sample "
                                    f"{sample['sample_id']}: {exc}"
                                )

                        if embeddings_added == 0:
                            if (
                                embedding_path
                                and os.path.exists(embedding_path)
                            ):
                                try:
                                    emb = np.load(
                                        embedding_path,
                                        allow_pickle=True,
                                    )
                                    if emb.dtype == object:
                                        emb = np.asarray(emb.tolist(), dtype=np.float32)

                                    from face_voice_rec.domain.face.face_core import (
                                        save_embedding_for_person,
                                    )

                                    save_embedding_for_person(
                                        selected_name,
                                        emb,
                                        image_path=best_image,
                                        quality=best_quality,
                                    )

                                    embeddings_added = 1

                                except Exception as exc:
                                    st.error(
                                        "Could not add the unknown "
                                        f"embedding: {exc}"
                                    )

                        if embeddings_added > 0:
                            resolve_unknown(
                                unknown_id,
                                selected_person_id,
                            )

                            get_face_index().rebuild()

                            st.success(
                                f"{label} assigned to "
                                f"{selected_name}. "
                                f"{embeddings_added} face sample"
                                f"{'s' if embeddings_added != 1 else ''} "
                                f"added to the face database."
                            )

                            st.rerun()

                        else:
                            st.error(
                                "No valid face embedding could be "
                                "added."
                            )

            else:
                st.info(
                    "No existing persons available. "
                    "Create a person below."
                )

            st.markdown("#### 🆕 Create New Person")

            new_name = st.text_input(
                "New person's name",
                key=f"new_person_name_{unknown_id}",
            )

            if st.button(
                "➕ Create New Person",
                key=f"create_person_{unknown_id}",
            ):
                if not new_name.strip():
                    st.warning(
                        "Please enter a name."
                    )

                else:
                    new_name = new_name.strip()

                    try:
                        new_person_id = create_person(
                            new_name
                        )

                        embeddings_added = 0

                        from face_voice_rec.domain.face.face_core import (
                            load_or_extract_embedding,
                            save_embedding_for_person,
                        )

                        for sample in sample_records:
                            try:
                                emb = load_or_extract_embedding(
                                    sample["embedding_path"],
                                    sample["image_path"],
                                )
                                if emb is None:
                                    raise ValueError("Could not extract a valid face embedding from sample image")

                                save_embedding_for_person(
                                    new_name,
                                    emb,
                                    image_path=sample["image_path"],
                                    quality=sample["quality"],
                                )

                                embeddings_added += 1

                            except Exception as exc:
                                st.warning(
                                    f"Could not add sample "
                                    f"{sample['sample_id']}: {exc}"
                                )

                        if embeddings_added == 0:
                            if (
                                embedding_path
                                and os.path.exists(embedding_path)
                            ):
                                emb = load_or_extract_embedding(
                                    embedding_path,
                                    best_image,
                                )
                                if emb is None:
                                    raise ValueError("Could not extract a valid face embedding from sample image")

                                save_embedding_for_person(
                                    new_name,
                                    emb,
                                    image_path=best_image,
                                    quality=best_quality,
                                )

                                embeddings_added = 1

                        if embeddings_added > 0:
                            resolve_unknown(
                                unknown_id,
                                new_person_id,
                            )

                            get_face_index().rebuild()

                            st.success(
                                f"Created {new_name} and added "
                                f"{embeddings_added} face sample"
                                f"{'s' if embeddings_added != 1 else ''}."
                            )

                            st.rerun()

                        else:
                            st.error(
                                "Person was created, but no valid "
                                "face embedding was available."
                            )

                    except Exception as exc:
                        st.exception(exc)
