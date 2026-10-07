"""Disk-backed temporal history preparation without model downloads."""

from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

import numpy as np

from multimodal_churn.data.temporal_store import (
    TemporalHistoryStore, build_history_arrays, observed_article_ids,
    stage_observed_transactions,
)


class TemporalStoreTest(unittest.TestCase):
    def test_cutoff_order_gaps_truncation_and_missing(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            archive_path = root / "hm.zip"
            lines = ["t_dat,customer_id,article_id,price,sales_channel_id"]
            for event in range(35):
                lines.append(f"2019-01-{event % 28 + 1:02d},c1,{event + 1:010d},0.1,1")
            lines += [
                "2019-01-01,c2,0000000099,0.2,2",
                "2019-01-04,c2,0000000100,0.3,1",
                "2019-02-01,c2,0000000101,0.4,2",
            ]
            with ZipFile(archive_path, "w") as archive:
                archive.writestr("hm/transactions_train.csv", "\n".join(lines) + "\n")
            db = root / "observed.sqlite"
            with ZipFile(archive_path) as archive:
                stats = stage_observed_transactions(archive, db, "2019-01-31", chunk_size=7)
            self.assertEqual(stats["observed_rows"], 37)
            self.assertNotIn("0000000101", observed_article_ids(db))
            self.assertNotIn("0000000099", observed_article_ids(db, max_customers=1))
            text = {"0000000099": np.ones(384, dtype="float32")}
            image = {"0000000100": np.ones(512, dtype="float32")}
            count = build_history_arrays(db, root / "histories", text, image, 32)
            self.assertEqual(count, 2)
            store = TemporalHistoryStore(root / "histories")
            self.assertEqual(store.customer_ids, ["c1", "c2"])
            self.assertEqual(int(store.arrays["transaction_original"][0]), 35)
            self.assertEqual(int(store.arrays["transaction_used"][0]), 32)
            values, valid = store.batch("transaction", np.array([1]))
            self.assertEqual(valid.tolist(), [[True, True] + [False] * 30])
            self.assertEqual(values[0, :2, 2].tolist(), [0.0, 3.0])
            text_values, text_mask = store.batch("text", np.array([0, 1]))
            image_values, image_mask = store.batch("image", np.array([0, 1]))
            self.assertEqual(text_mask.sum(axis=1).tolist(), [0, 1])
            self.assertEqual(image_mask.sum(axis=1).tolist(), [0, 1])
            self.assertEqual(text_values.shape, (2, 32, 384))
            self.assertEqual(image_values.shape, (2, 32, 512))
            store.close()
            build_history_arrays(db, root / "empty_products", {}, {}, 32)
            empty = TemporalHistoryStore(root / "empty_products")
            _, empty_mask = empty.batch("image", np.array([0, 1]))
            self.assertFalse(empty_mask.any())
            empty.close()


if __name__ == "__main__":
    unittest.main()
