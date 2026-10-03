"""CHiLS sub-concept classifier with group-max class scores."""
import torch
import torch.nn as nn


class SubConceptClf(nn.Module):
    """CLIP visual tower with group-max sub-concept scores."""

    def __init__(self, visual, text_w, group_id, n_classes, scale=100.0):
        super().__init__()
        self.visual = visual
        self.head = nn.Parameter(text_w.clone())
        self.register_buffer("group_id", group_id.clone())
        self.n_classes = n_classes
        self.scale = scale
        m = torch.full((n_classes, text_w.shape[0]), float("-inf"))
        m[group_id, torch.arange(text_w.shape[0])] = 0.0
        self.register_buffer("mask", m)

    def forward(self, x):
        f = self.visual(x)
        f = f / f.norm(dim=-1, keepdim=True)
        w = self.head / self.head.norm(dim=-1, keepdim=True)
        sim = self.scale * (f @ w.T)
        return (sim.unsqueeze(1) + self.mask.unsqueeze(0)).amax(dim=2)


def build_subconcept_head(clip, tok, classes, names_map, templates):
    """Build sub-concept text embeddings."""
    rows, gid, flat = [], [], []
    with torch.no_grad():
        for ci, c in enumerate(classes):
            for n in names_map[c]:
                e = clip.encode_text(tok([t.format(n) for t in templates]))
                e = e / e.norm(dim=-1, keepdim=True)
                v = e.mean(0)
                rows.append(v / v.norm())
                gid.append(ci)
                flat.append(n)
    return torch.stack(rows), torch.tensor(gid, dtype=torch.long), flat


if __name__ == "__main__":
    S, D, C = 117, 64, 20
    w = torch.randn(S, D); w = w / w.norm(dim=-1, keepdim=True)
    gid = torch.arange(S) % C
    m = SubConceptClf(nn.Identity(), w, gid, C)
    x = torch.randn(8, D)
    out = m(x)
    print("group-max head OK:", tuple(out.shape), "finite:", bool(torch.isfinite(out).all()))
    print("classes with no sub-concept:", int((torch.bincount(gid, minlength=C) == 0).sum()))
