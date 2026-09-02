"""Training and serving must preprocess identically.

A preprocessing difference between training and inference does not raise; it
degrades accuracy in a way that looks like a modelling problem and sends you
looking in the wrong place. This project has already paid for one such
mismatch, and nearly paid for a second: SmolVLM's processor defaults to a
longest edge of 2048 while training ran at 512, so serving would have upscaled
every image fourfold against the weights it was fed.

These tests pin the shared values rather than checking they happen to agree
today.
"""
import satquery.adapt.train_vlm as trainer
import satquery.tools.generative as serving


def test_prompts_come_from_one_definition():
    assert trainer.PROMPTS is serving.PROMPTS


def test_image_size_comes_from_one_definition():
    assert trainer.IMAGE_SIZE == serving.IMAGE_SIZE


def test_base_model_comes_from_one_definition():
    assert trainer.BASE == serving.BASE


def test_every_task_has_a_prompt():
    """A task with no entry silently falls back to the bare question, which is
    a different distribution from the one it was trained on."""
    for task in ("vqa", "caption", "refer"):
        assert task in serving.PROMPTS
        assert "{q}" in serving.PROMPTS[task]


def test_image_size_is_not_the_processor_default():
    """If these ever coincide the parity test above stops proving anything, so
    the value is asserted explicitly."""
    assert serving.IMAGE_SIZE == 512
