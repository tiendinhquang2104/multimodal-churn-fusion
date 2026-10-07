"""Small local checks for H&M sample loading and feature alignment."""

from io import BytesIO
import unittest
from zipfile import ZipFile

import numpy as np

from multimodal_churn.data.features import (
    concat_feature_table,
    mean_purchased_article_vectors,
    random_projection,
)
from multimodal_churn.data.hm import (
    hm_image_members,
    read_hm_articles,
    read_hm_transactions,
)
from multimodal_churn.data.transaction import transaction_sequences


class FeatureTableDataTest(unittest.TestCase):
    """Verify one customer row and no purchases after the observation cutoff."""

    def test_zip_to_aligned_blocks(self) -> None:
        archive_bytes = BytesIO()
        with ZipFile(archive_bytes, "w") as archive:
            archive.writestr(
                "hm/transactions_train.csv",
                "t_dat,customer_id,article_id,price,sales_channel_id\n"
                "2019-01-01,c1,0000000001,0.10,1\n"
                "2019-01-02,c2,0000000002,0.20,2\n"
                "2019-01-03,c1,0000000003,0.30,2\n"
                "2019-02-01,c1,0000000004,0.40,1\n",
            )
            archive.writestr(
                "hm/articles.csv",
                "article_id,prod_name,detail_desc\n"
                "0000000001,Top,Blue\n"
                "0000000002,Shirt,Red\n"
                "0000000003,Coat,Black\n",
            )
            archive.writestr("hm/images/000/0000000001.jpg", b"placeholder")
        archive_bytes.seek(0)
        with ZipFile(archive_bytes) as archive:
            transactions = read_hm_transactions(
                archive,
                observation_end="2019-01-31",
                max_customers=2,
                max_sequence_length=2,
                chunk_size=2,
            )
            articles = read_hm_articles(
                archive, set(transactions["article_id"].astype(str))
            )
            images = hm_image_members(
                archive, set(transactions["article_id"].astype(str))
            )
        self.assertEqual(len(transactions), 3)
        self.assertNotIn("0000000004", set(transactions["article_id"].astype(str)))
        self.assertEqual(len(articles), 3)
        self.assertEqual(set(images), {"0000000001"})

        customer_ids, values, mask = transaction_sequences(
            transactions, observation_end="2019-01-31", max_sequence_length=2
        )
        self.assertEqual(customer_ids, ["c1", "c2"])
        self.assertEqual(values.shape, (2, 2, 3))
        self.assertEqual(mask.tolist(), [[True, True], [True, False]])

        vectors = {
            "0000000001": np.array([1.0, 0.0], dtype=np.float32),
            "0000000002": np.array([0.0, 1.0], dtype=np.float32),
        }
        pooled, available = mean_purchased_article_vectors(
            transactions, customer_ids, vectors
        )
        self.assertEqual(available.tolist(), [True, True])
        np.testing.assert_allclose(pooled, [[1, 0], [0, 1]])
        projected, matrix = random_projection(pooled, output_dim=2, seed=42)
        np.testing.assert_allclose(projected, pooled @ matrix)
        table = concat_feature_table(customer_ids, values[:, 0, :2], projected, pooled)
        self.assertEqual(table.shape, (2, 7))
        self.assertEqual(
            table.columns.tolist(),
            [
                "customer_id",
                "transaction_000",
                "transaction_001",
                "text_000",
                "text_001",
                "image_000",
                "image_001",
            ],
        )
        self.assertNotIn("label", table.columns)


if __name__ == "__main__":
    unittest.main()
