from figma_rag.ingestion.help_center import (
    ArticleMetadata,
    build_document_id,
    is_article_in_scope,
    is_help_center_article_url,
    normalize_url,
    parse_sitemap_xml,
)


def test_normalize_url_removes_query_and_fragment() -> None:
    assert (
        normalize_url("https://help.figma.com/hc/en-us/articles/123-guide?x=1#top")
        == "https://help.figma.com/hc/en-us/articles/123-guide"
    )


def test_article_url_filter_accepts_only_english_article_paths() -> None:
    assert is_help_center_article_url("https://help.figma.com/hc/en-us/articles/123-guide")
    assert not is_help_center_article_url("https://help.figma.com/hc/fr/articles/123-guide")
    assert not is_help_center_article_url("https://help.figma.com/hc/en-us/categories/123-guide")
    assert not is_help_center_article_url("https://developers.figma.com/docs/example")


def test_scope_filter_excludes_dev_mode_articles() -> None:
    metadata = ArticleMetadata(
        document_id="help-center-123-dev-mode-basics",
        title="Dev Mode basics",
        source_url="https://help.figma.com/hc/en-us/articles/123-dev-mode-basics",
        source_type="help_center",
        product_area="figma-design",
        breadcrumbs=("Figma Design", "Dev Mode"),
    )

    assert is_article_in_scope(metadata) == (False, "dev mode")


def test_scope_filter_excludes_work_together_in_files_articles() -> None:
    metadata = ArticleMetadata(
        document_id="help-center-456-collaboration",
        title="Leave comments in files",
        source_url="https://help.figma.com/hc/en-us/articles/456-leave-comments",
        source_type="help_center",
        product_area="figma-design",
        breadcrumbs=("Figma Design", "Work Together in Files"),
    )

    assert is_article_in_scope(metadata) == (False, "work together in files")


def test_build_document_id_uses_article_slug() -> None:
    assert (
        build_document_id(
            "https://help.figma.com/hc/en-us/articles/123456789-auto-layout-basics",
            "Auto layout basics",
        )
        == "help-center-123456789-auto-layout-basics"
    )


def test_parse_sitemap_index_returns_nested_sitemaps() -> None:
    xml_content = b"""<?xml version="1.0" encoding="UTF-8"?>
    <sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <sitemap>
        <loc>https://help.figma.com/hc/sitemap_1.xml</loc>
      </sitemap>
    </sitemapindex>
    """

    sitemap_urls, page_urls = parse_sitemap_xml(xml_content)

    assert sitemap_urls == ["https://help.figma.com/hc/sitemap_1.xml"]
    assert page_urls == []


def test_parse_urlset_returns_page_urls() -> None:
    xml_content = b"""<?xml version="1.0" encoding="UTF-8"?>
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <url>
        <loc>https://help.figma.com/hc/en-us/articles/123-guide</loc>
      </url>
    </urlset>
    """

    sitemap_urls, page_urls = parse_sitemap_xml(xml_content)

    assert sitemap_urls == []
    assert page_urls == ["https://help.figma.com/hc/en-us/articles/123-guide"]
