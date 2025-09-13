"""
Enhanced data sources module with multiple free APIs and real-time verification.
"""

import asyncio
import aiohttp
import feedparser
import requests
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from urllib.parse import urljoin, urlparse, quote_plus
import re
import time

# Document processing
try:
    from newspaper import Article
    import PyPDF2
    from docx import Document
    from bs4 import BeautifulSoup
    DOCUMENT_PROCESSING = True
except ImportError:
    DOCUMENT_PROCESSING = False

from .config import DATA_SOURCES, settings
from .utils import cache_result, clean_text, extract_keywords, retry_on_failure, logger

class EnhancedDataSourceManager:
    """Enhanced multi-source data manager with real-time verification."""
    
    def __init__(self):
        self.session = None
        self.rate_limiter = {}
        self.source_reliability = {}
        self._setup_session()
        self._initialize_free_apis()
    
    def _setup_session(self):
        """Setup HTTP session with proper headers."""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/html, application/xhtml+xml, application/xml;q=0.9, image/webp, */*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0'
        })
    
    def _initialize_free_apis(self):
        """Initialize free API configurations."""
        self.free_apis = {
            "newsapi": {
                "url": "https://newsapi.org/v2/everything",
                "key_param": "apiKey",
                "daily_limit": 500,
                "reliability": 0.9
            },
            "newsdata": {
                "url": "https://newsdata.io/api/1/news",
                "key_param": "apikey", 
                "daily_limit": 200,
                "reliability": 0.85
            },
            "mediastack": {
                "url": "http://api.mediastack.com/v1/news",
                "key_param": "access_key",
                "monthly_limit": 500,
                "reliability": 0.8
            },
            "currents": {
                "url": "https://api.currentsapi.services/v1/search",
                "key_param": "apiKey",
                "daily_limit": 600,
                "reliability": 0.75
            },
            "gnews": {
                "url": "https://gnews.io/api/v4/search",
                "key_param": "apikey",
                "daily_limit": 100,
                "reliability": 0.8
            },
            "worldnews": {
                "url": "https://api.worldnewsapi.com/search-news",
                "key_param": "api-key",
                "daily_limit": 100,
                "reliability": 0.7
            }
        }
        
        # Enhanced RSS sources with reliability scores
        self.enhanced_rss_sources = [
            ("BBC News", "http://feeds.bbci.co.uk/news/rss.xml", 0.95),
            ("Reuters", "http://feeds.reuters.com/reuters/topNews", 0.95),
            ("Associated Press", "https://feeds.apnews.com/rss/apf-topnews", 0.95),
            ("NPR", "https://feeds.npr.org/1001/rss.xml", 0.9),
            ("The Guardian", "https://www.theguardian.com/world/rss", 0.85),
            ("CNN", "http://rss.cnn.com/rss/edition.rss", 0.8),
            ("ABC News", "https://abcnews.go.com/abcnews/topstories", 0.85),
            ("CBS News", "https://www.cbsnews.com/latest/rss/main", 0.8),
            ("NBC News", "https://feeds.nbcnews.com/nbcnews/public/news", 0.8),
            ("Washington Post", "https://feeds.washingtonpost.com/rss/national", 0.85),
            ("New York Times", "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml", 0.9),
            ("Wall Street Journal", "https://feeds.a.dj.com/rss/RSSWorldNews.xml", 0.9),
            ("Financial Times", "https://www.ft.com/rss/home", 0.85),
            ("Bloomberg", "https://feeds.bloomberg.com/politics/news.rss", 0.85),
            ("Al Jazeera", "https://www.aljazeera.com/xml/rss/all.xml", 0.8)
        ]
    
    async def get_real_time_news_verification(self, text: str, depth: str = "comprehensive") -> Dict[str, Any]:
        """
        Get real-time news verification with visual source tracking.
        
        Args:
            text: Text to verify
            depth: Analysis depth
            
        Returns:
            Comprehensive verification with real-time sources
        """
        verification_start_time = time.time()
        keywords = extract_keywords(text, 8)
        
        # Create verification session
        verification_session = {
            "session_id": f"verify_{int(time.time())}",
            "query": text[:100] + "..." if len(text) > 100 else text,
            "keywords": keywords,
            "started_at": datetime.now().isoformat(),
            "sources_checked": [],
            "real_time_results": [],
            "verification_status": "in_progress"
        }
        
        # Concurrent verification across all sources
        tasks = []
        
        # Free API sources
        for api_name, config in self.free_apis.items():
            if self._should_use_api(api_name):
                tasks.append(self._verify_with_free_api(api_name, config, keywords))
        
        # Enhanced RSS verification
        tasks.append(self._enhanced_rss_verification(keywords))
        
        # Wikipedia and knowledge bases
        tasks.append(self._wikipedia_verification(keywords))
        
        # Social media mentions (free tiers)
        tasks.append(self._social_media_verification(keywords))
        
        # Internet Archive verification
        tasks.append(self._internet_archive_verification(keywords))
        
        # Execute all verifications concurrently
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            processed_results = self._process_real_time_results(results, verification_session)
            
            # Calculate real-time trust score
            trust_score = self._calculate_real_time_trust_score(processed_results)
            
            # Generate verification summary
            verification_summary = self._generate_verification_summary(processed_results, trust_score)
            
            processing_time = time.time() - verification_start_time
            
            return {
                "session": verification_session,
                "processing_time": processing_time,
                "trust_score": trust_score,
                "verification_summary": verification_summary,
                "detailed_results": processed_results,
                "real_time_sources": self._get_source_status(),
                "recommendation": self._generate_recommendation(trust_score)
            }
            
        except Exception as e:
            logger.error(f"Real-time verification failed: {e}")
            return {
                "error": str(e),
                "session": verification_session,
                "fallback_verification": await self._fallback_verification(keywords)
            }
    
    async def _verify_with_free_api(self, api_name: str, config: Dict, keywords: List[str]) -> Dict[str, Any]:
        """Verify with specific free API."""
        try:
            # Get API key from environment or use demo mode
            api_key = getattr(settings, f"{api_name}_key", "")
            
            if not api_key:
                return {
                    "source": api_name,
                    "status": "no_api_key",
                    "message": f"{api_name} requires API key for enhanced verification"
                }
            
            query = " ".join(keywords[:3])
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
                params = {
                    config["key_param"]: api_key,
                    "q": query,
                    "language": "en",
                    "sortBy": "relevancy" if api_name == "newsapi" else "published_desc",
                    "pageSize": 20 if api_name == "newsapi" else None,
                    "size": 20 if api_name == "newsdata" else None,
                    "limit": 20 if api_name in ["currents", "gnews"] else None
                }
                
                # Remove None values
                params = {k: v for k, v in params.items() if v is not None}
                
                async with session.get(config["url"], params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        articles = []
                        if api_name == "newsapi":
                            articles = data.get("articles", [])
                        elif api_name == "newsdata":
                            articles = data.get("results", [])
                        elif api_name == "mediastack":
                            articles = data.get("data", [])
                        elif api_name == "currents":
                            articles = data.get("news", [])
                        elif api_name == "gnews":
                            articles = data.get("articles", [])
                        elif api_name == "worldnews":
                            articles = data.get("news", [])
                        
                        return {
                            "source": api_name,
                            "status": "success",
                            "reliability_score": config["reliability"],
                            "articles_found": len(articles),
                            "articles": self._clean_articles(articles, api_name)[:10],
                            "query_used": query,
                            "api_response_time": response.headers.get("X-Response-Time", "unknown")
                        }
                    else:
                        return {
                            "source": api_name,
                            "status": "api_error",
                            "error_code": response.status,
                            "message": f"API returned {response.status}"
                        }
        
        except Exception as e:
            return {
                "source": api_name,
                "status": "error",
                "error": str(e)
            }
    
    async def _enhanced_rss_verification(self, keywords: List[str]) -> Dict[str, Any]:
        """Enhanced RSS verification with reliability scoring."""
        try:
            all_articles = []
            source_results = {}
            keywords_lower = [k.lower() for k in keywords]
            
            for source_name, feed_url, reliability in self.enhanced_rss_sources:
                try:
                    feed = feedparser.parse(feed_url)
                    
                    if not feed.entries:
                        continue
                    
                    source_articles = []
                    for entry in feed.entries[:20]:
                        title = getattr(entry, 'title', '').lower()
                        summary = getattr(entry, 'summary', '').lower()
                        description = getattr(entry, 'description', '').lower()
                        
                        content_text = f"{title} {summary} {description}"
                        
                        # Calculate relevance score
                        relevance_score = sum(1 for keyword in keywords_lower if keyword in content_text) / len(keywords_lower)
                        
                        if relevance_score > 0.1:  # At least 10% keyword match
                            article = {
                                "title": getattr(entry, 'title', 'No title'),
                                "link": getattr(entry, 'link', '#'),
                                "source": source_name,
                                "published": getattr(entry, 'published', 'Unknown date'),
                                "summary": (summary or description)[:300] + "..." if (summary or description) else "No summary",
                                "relevance_score": relevance_score,
                                "source_reliability": reliability
                            }
                            source_articles.append(article)
                            
                            if len(source_articles) >= 5:
                                break
                    
                    if source_articles:
                        source_results[source_name] = {
                            "articles": source_articles,
                            "reliability": reliability,
                            "articles_count": len(source_articles)
                        }
                        all_articles.extend(source_articles)
                        
                except Exception as e:
                    logger.warning(f"RSS feed {source_name} failed: {e}")
                    continue
            
            return {
                "source": "enhanced_rss",
                "status": "success",
                "total_articles": len(all_articles),
                "sources_checked": len(source_results),
                "source_breakdown": source_results,
                "top_articles": sorted(all_articles, key=lambda x: x["relevance_score"], reverse=True)[:15]
            }
            
        except Exception as e:
            return {
                "source": "enhanced_rss",
                "status": "error",
                "error": str(e)
            }
    
    async def _wikipedia_verification(self, keywords: List[str]) -> Dict[str, Any]:
        """Enhanced Wikipedia verification."""
        try:
            verification_results = []
            
            for keyword in keywords[:3]:  # Check top 3 keywords
                # Search Wikipedia
                search_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote_plus(keyword)}"
                
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                    async with session.get(search_url) as response:
                        if response.status == 200:
                            data = await response.json()
                            if 'extract' in data and len(data['extract']) > 50:
                                verification_results.append({
                                    "keyword": keyword,
                                    "title": data.get('title', 'Unknown'),
                                    "extract": data.get('extract', '')[:500] + "...",
                                    "url": data.get('content_urls', {}).get('desktop', {}).get('page', ''),
                                    "type": data.get('type', 'standard'),
                                    "reliability_score": 0.9  # Wikipedia is highly reliable
                                })
            
            # Also search Wikidata for structured data
            wikidata_results = await self._search_wikidata(keywords[:2])
            
            return {
                "source": "wikipedia",
                "status": "success",
                "wikipedia_results": verification_results,
                "wikidata_results": wikidata_results,
                "total_matches": len(verification_results) + len(wikidata_results)
            }
            
        except Exception as e:
            return {
                "source": "wikipedia",
                "status": "error", 
                "error": str(e)
            }
    
    async def _search_wikidata(self, keywords: List[str]) -> List[Dict[str, Any]]:
        """Search Wikidata for structured information."""
        try:
            results = []
            
            for keyword in keywords:
                search_url = f"https://www.wikidata.org/w/api.php"
                params = {
                    "action": "wbsearchentities",
                    "search": keyword,
                    "language": "en",
                    "format": "json",
                    "limit": 3
                }
                
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=8)) as session:
                    async with session.get(search_url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            entities = data.get('search', [])
                            
                            for entity in entities:
                                results.append({
                                    "keyword": keyword,
                                    "entity_id": entity.get('id'),
                                    "label": entity.get('label', ''),
                                    "description": entity.get('description', ''),
                                    "url": f"https://www.wikidata.org/wiki/{entity.get('id')}"
                                })
            
            return results
            
        except Exception as e:
            logger.warning(f"Wikidata search failed: {e}")
            return []
    
    async def _social_media_verification(self, keywords: List[str]) -> Dict[str, Any]:
        """Verify against social media mentions (free tiers)."""
        try:
            # Reddit verification (free)
            reddit_results = await self._search_reddit(keywords)
            
            # Twitter mentions would require API key, so we'll simulate the structure
            social_results = {
                "reddit": reddit_results,
                "twitter": {"status": "requires_api_key", "message": "Twitter API key needed for social verification"},
                "total_mentions": len(reddit_results.get("posts", []))
            }
            
            return {
                "source": "social_media",
                "status": "partial_success",
                "results": social_results
            }
            
        except Exception as e:
            return {
                "source": "social_media",
                "status": "error",
                "error": str(e)
            }
    
    async def _search_reddit(self, keywords: List[str]) -> Dict[str, Any]:
        """Search Reddit for mentions."""
        try:
            query = " ".join(keywords[:2])
            reddit_url = f"https://www.reddit.com/search.json"
            params = {
                "q": query,
                "sort": "relevance",
                "limit": 10,
                "type": "link"
            }
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get(reddit_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        posts = []
                        
                        for post in data.get('data', {}).get('children', []):
                            post_data = post.get('data', {})
                            posts.append({
                                "title": post_data.get('title', ''),
                                "subreddit": post_data.get('subreddit', ''),
                                "score": post_data.get('score', 0),
                                "url": f"https://reddit.com{post_data.get('permalink', '')}",
                                "created_utc": post_data.get('created_utc', 0)
                            })
                        
                        return {
                            "status": "success",
                            "posts": posts,
                            "query_used": query
                        }
            
            return {"status": "no_results"}
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def _internet_archive_verification(self, keywords: List[str]) -> Dict[str, Any]:
        """Search Internet Archive for historical context."""
        try:
            query = " ".join(keywords[:2])
            archive_url = "https://archive.org/advancedsearch.php"
            params = {
                "q": query,
                "fl": "identifier,title,description,date,creator",
                "rows": 10,
                "page": 1,
                "output": "json"
            }
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get(archive_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        docs = data.get('response', {}).get('docs', [])
                        
                        results = []
                        for doc in docs:
                            results.append({
                                "identifier": doc.get('identifier'),
                                "title": doc.get('title', ''),
                                "description": doc.get('description', ''),
                                "date": doc.get('date', ''),
                                "creator": doc.get('creator', ''),
                                "url": f"https://archive.org/details/{doc.get('identifier')}"
                            })
                        
                        return {
                            "source": "internet_archive",
                            "status": "success",
                            "results": results,
                            "total_found": len(results)
                        }
            
            return {
                "source": "internet_archive", 
                "status": "no_results"
            }
            
        except Exception as e:
            return {
                "source": "internet_archive",
                "status": "error",
                "error": str(e)
            }
    
    def _clean_articles(self, articles: List[Dict], api_name: str) -> List[Dict[str, Any]]:
        """Clean and normalize articles from different APIs."""
        cleaned = []
        
        for article in articles:
            try:
                # Different APIs have different structures
                if api_name == "newsapi":
                    cleaned_article = {
                        "title": article.get('title', 'No title')[:200],
                        "description": article.get('description', '')[:500],
                        "url": article.get('url', ''),
                        "source": article.get('source', {}).get('name', 'Unknown'),
                        "published_at": article.get('publishedAt', ''),
                        "author": article.get('author', 'Unknown')
                    }
                elif api_name == "newsdata":
                    cleaned_article = {
                        "title": article.get('title', 'No title')[:200],
                        "description": article.get('description', '')[:500],
                        "url": article.get('link', ''),
                        "source": article.get('source_id', 'Unknown'),
                        "published_at": article.get('pubDate', ''),
                        "author": article.get('creator', ['Unknown'])[0] if article.get('creator') else 'Unknown'
                    }
                elif api_name == "mediastack":
                    cleaned_article = {
                        "title": article.get('title', 'No title')[:200],
                        "description": article.get('description', '')[:500],
                        "url": article.get('url', ''),
                        "source": article.get('source', 'Unknown'),
                        "published_at": article.get('published_at', ''),
                        "author": article.get('author', 'Unknown')
                    }
                elif api_name == "currents":
                    cleaned_article = {
                        "title": article.get('title', 'No title')[:200],
                        "description": article.get('description', '')[:500],
                        "url": article.get('url', ''),
                        "source": article.get('author', 'Unknown'),
                        "published_at": article.get('published', ''),
                        "author": article.get('author', 'Unknown')
                    }
                elif api_name == "gnews":
                    cleaned_article = {
                        "title": article.get('title', 'No title')[:200],
                        "description": article.get('description', '')[:500],
                        "url": article.get('url', ''),
                        "source": article.get('source', {}).get('name', 'Unknown'),
                        "published_at": article.get('publishedAt', ''),
                        "author": 'Unknown'
                    }
                elif api_name == "worldnews":
                    cleaned_article = {
                        "title": article.get('title', 'No title')[:200],
                        "description": article.get('text', '')[:500],
                        "url": article.get('url', ''),
                        "source": article.get('source_country', 'Unknown'),
                        "published_at": article.get('publish_date', ''),
                        "author": article.get('author', 'Unknown')
                    }
                else:
                    # Fallback structure
                    cleaned_article = {
                        "title": str(article.get('title', 'No title'))[:200],
                        "description": str(article.get('description', ''))[:500],
                        "url": article.get('url', ''),
                        "source": str(article.get('source', 'Unknown')),
                        "published_at": article.get('published_at', ''),
                        "author": str(article.get('author', 'Unknown'))
                    }
                
                # Add reliability scoring
                cleaned_article['api_source'] = api_name
                cleaned_article['reliability_score'] = self.free_apis.get(api_name, {}).get('reliability', 0.5)
                
                cleaned.append(cleaned_article)
                
            except Exception as e:
                logger.warning(f"Failed to clean article from {api_name}: {e}")
                continue
        
        return cleaned
    
    def _process_real_time_results(self, results: List[Any], session: Dict) -> Dict[str, Any]:
        """Process real-time verification results."""
        processed = {
            "successful_sources": [],
            "failed_sources": [],
            "total_articles": 0,
            "source_breakdown": {},
            "reliability_scores": {},
            "real_time_status": [],
            "verification_timeline": []
        }
        
        for result in results:
            if isinstance(result, Exception):
                processed["failed_sources"].append(str(result))
                continue
            
            if not isinstance(result, dict):
                continue
            
            source_name = result.get("source", "unknown")
            status = result.get("status", "unknown")
            
            # Track real-time status
            processed["real_time_status"].append({
                "source": source_name,
                "status": status,
                "timestamp": datetime.now().isoformat(),
                "reliability": result.get("reliability_score", 0.5)
            })
            
            if status == "success":
                processed["successful_sources"].append(source_name)
                
                # Count articles
                if "articles_found" in result:
                    processed["total_articles"] += result["articles_found"]
                elif "total_articles" in result:
                    processed["total_articles"] += result["total_articles"]
                elif "total_matches" in result:
                    processed["total_articles"] += result["total_matches"]
                
                # Store source breakdown
                processed["source_breakdown"][source_name] = result
                
                # Store reliability score
                processed["reliability_scores"][source_name] = result.get("reliability_score", 0.5)
            
            else:
                error_msg = result.get("error", result.get("message", "Unknown error"))
                processed["failed_sources"].append(f"{source_name}: {error_msg}")
        
        return processed
    
    def _calculate_real_time_trust_score(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate real-time trust score."""
        successful_sources = len(results["successful_sources"])
        total_articles = results["total_articles"]
        reliability_scores = results["reliability_scores"]
        
        if not successful_sources:
            return {
                "overall_score": 0.0,
                "confidence_level": "no_verification",
                "factors": {
                    "source_diversity": 0.0,
                    "article_coverage": 0.0,
                    "source_reliability": 0.0,
                    "verification_depth": 0.0
                }
            }
        
        # Factor 1: Source diversity (0-1)
        source_diversity = min(successful_sources / 8, 1.0)  # Normalize to 8 sources
        
        # Factor 2: Article coverage (0-1)
        article_coverage = min(total_articles / 50, 1.0)  # Normalize to 50 articles
        
        # Factor 3: Average source reliability (0-1)
        if reliability_scores:
            avg_reliability = sum(reliability_scores.values()) / len(reliability_scores)
        else:
            avg_reliability = 0.0
        
        # Factor 4: Verification depth (0-1)
        verification_depth = min(len(results["source_breakdown"]) / 5, 1.0)
        
        # Weighted combination
        weights = {
            "source_diversity": 0.3,
            "article_coverage": 0.25,
            "source_reliability": 0.3,
            "verification_depth": 0.15
        }
        
        overall_score = (
            weights["source_diversity"] * source_diversity +
            weights["article_coverage"] * article_coverage +
            weights["source_reliability"] * avg_reliability +
            weights["verification_depth"] * verification_depth
        )
        
        # Determine confidence level
        if overall_score >= 0.8:
            confidence_level = "high"
        elif overall_score >= 0.6:
            confidence_level = "medium"
        elif overall_score >= 0.3:
            confidence_level = "low"
        else:
            confidence_level = "very_low"
        
        return {
            "overall_score": overall_score,
            "confidence_level": confidence_level,
            "factors": {
                "source_diversity": source_diversity,
                "article_coverage": article_coverage,
                "source_reliability": avg_reliability,
                "verification_depth": verification_depth
            },
            "weights": weights,
            "breakdown": {
                "successful_sources": successful_sources,
                "total_articles": total_articles,
                "avg_reliability": avg_reliability
            }
        }
    
    def _generate_verification_summary(self, results: Dict[str, Any], trust_score: Dict[str, Any]) -> str:
        """Generate human-readable verification summary."""
        successful = len(results["successful_sources"])
        total_articles = results["total_articles"]
        confidence = trust_score["confidence_level"]
        score = trust_score["overall_score"]
        
        if confidence == "high":
            summary = f"✅ **HIGH CONFIDENCE VERIFICATION** ({score:.1%})\n"
            summary += f"Successfully verified across **{successful} trusted sources** with **{total_articles} related articles** found. "
            summary += "Multiple independent sources confirm the information with high reliability."
        
        elif confidence == "medium":
            summary = f"⚠️ **MODERATE CONFIDENCE VERIFICATION** ({score:.1%})\n"
            summary += f"Verified across **{successful} sources** with **{total_articles} articles** found. "
            summary += "Some verification available but additional sources recommended."
        
        elif confidence == "low":
            summary = f"🔍 **LIMITED VERIFICATION** ({score:.1%})\n"
            summary += f"Found **{successful} sources** with **{total_articles} articles**. "
            summary += "Limited verification available - exercise caution and seek additional sources."
        
        else:
            summary = f"❌ **NO RELIABLE VERIFICATION** ({score:.1%})\n"
            summary += f"Minimal verification found from **{successful} sources**. "
            summary += "Information cannot be independently confirmed - high risk of misinformation."
        
        # Add source breakdown
        if results["successful_sources"]:
            summary += f"\n\n**Sources checked:** {', '.join(results['successful_sources'])}"
        
        if results["failed_sources"]:
            summary += f"\n**Note:** {len(results['failed_sources'])} sources unavailable"
        
        return summary
    
    def _generate_recommendation(self, trust_score: Dict[str, Any]) -> Dict[str, str]:
        """Generate actionable recommendations."""
        confidence = trust_score["confidence_level"]
        score = trust_score["overall_score"]
        
        if confidence == "high":
            return {
                "action": "SHARE_CAUTIOUSLY",
                "color": "success",
                "icon": "✅",
                "message": "Information appears to be well-verified by multiple trusted sources. Safe to share with confidence.",
                "details": "High reliability score with good source diversity and coverage."
            }
        
        elif confidence == "medium":
            return {
                "action": "VERIFY_FURTHER", 
                "color": "warning",
                "icon": "⚠️",
                "message": "Information has moderate verification. Consider checking additional sources before sharing.",
                "details": "Some verification available but could benefit from additional confirmation."
            }
        
        elif confidence == "low":
            return {
                "action": "EXERCISE_CAUTION",
                "color": "warning", 
                "icon": "🔍",
                "message": "Limited verification found. Exercise caution and seek additional reliable sources.",
                "details": "Low verification coverage - information may be incomplete or inaccurate."
            }
        
        else:
            return {
                "action": "DO_NOT_SHARE",
                "color": "danger",
                "icon": "❌", 
                "message": "Information cannot be reliably verified. Do not share without additional confirmation.",
                "details": "Insufficient verification - high risk of misinformation."
            }
    
    def _should_use_api(self, api_name: str) -> bool:
        """Check if API should be used based on rate limits and availability."""
        # In production, this would check actual rate limits
        # For demo, we'll assume all APIs are available
        return True
    
    def _get_source_status(self) -> List[Dict[str, Any]]:
        """Get current status of all sources."""
        statuses = []
        
        # API sources
        for api_name, config in self.free_apis.items():
            api_key = getattr(settings, f"{api_name}_key", "")
            statuses.append({
                "name": api_name.title(),
                "type": "api",
                "status": "active" if api_key else "needs_key",
                "reliability": config["reliability"],
                "daily_limit": config.get("daily_limit", 0)
            })
        
        # RSS sources
        for source_name, _, reliability in self.enhanced_rss_sources[:10]:
            statuses.append({
                "name": source_name,
                "type": "rss",
                "status": "active",
                "reliability": reliability,
                "daily_limit": "unlimited"
            })
        
        return statuses
    
    async def _fallback_verification(self, keywords: List[str]) -> Dict[str, Any]:
        """Fallback verification when main process fails."""
        try:
            # Use only RSS and Wikipedia as fallback
            rss_result = await self._enhanced_rss_verification(keywords)
            wikipedia_result = await self._wikipedia_verification(keywords)
            
            return {
                "fallback_mode": True,
                "rss_verification": rss_result,
                "wikipedia_verification": wikipedia_result,
                "message": "Using fallback verification methods"
            }
            
        except Exception as e:
            return {
                "fallback_mode": True,
                "error": str(e),
                "message": "All verification methods failed"
            }
    
    @cache_result("url_extraction", ttl=3600)
    def extract_article_from_url(self, url: str) -> Dict[str, Any]:
        """
        Enhanced URL extraction with better error handling and preview.
        
        Args:
            url: Article URL
            
        Returns:
            Extracted article data with preview
        """
        if not url or not url.startswith(('http://', 'https://')):
            return {"error": "Invalid URL format"}
        
        try:
            # Method 1: Try newspaper3k first
            if DOCUMENT_PROCESSING:
                try:
                    article = Article(url)
                    article.download()
                    article.parse()
                    
                    # Try to extract additional metadata
                    try:
                        article.nlp()
                        keywords = article.keywords
                        summary = article.summary
                    except:
                        keywords = []
                        summary = article.text[:500] + "..." if len(article.text) > 500 else article.text
                    
                    return {
                        "success": True,
                        "method": "newspaper3k",
                        "title": article.title,
                        "text": article.text,
                        "authors": article.authors,
                        "publish_date": article.publish_date.isoformat() if article.publish_date else None,
                        "top_image": article.top_image,
                        "url": url,
                        "summary": summary,
                        "keywords": keywords,
                        "word_count": len(article.text.split()),
                        "meta_description": getattr(article, 'meta_description', ''),
                        "meta_keywords": getattr(article, 'meta_keywords', ''),
                        "canonical_link": getattr(article, 'canonical_link', url)
                    }
                    
                except Exception as e:
                    logger.warning(f"Newspaper3k failed for {url}: {e}")
            
            # Method 2: Fallback to requests + BeautifulSoup
            try:
                response = self.session.get(url, timeout=15)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract title
                title = ""
                if soup.title:
                    title = soup.title.string.strip()
                elif soup.find('h1'):
                    title = soup.find('h1').get_text().strip()
                
                # Extract main content
                content_selectors = [
                    'article',
                    '[role="main"]',
                    '.post-content',
                    '.entry-content', 
                    '.article-content',
                    '.content',
                    'main'
                ]
                
                text = ""
                for selector in content_selectors:
                    content_elem = soup.select_one(selector)
                    if content_elem:
                        text = content_elem.get_text().strip()
                        break
                
                # If no content found, get all paragraphs
                if not text:
                    paragraphs = soup.find_all('p')
                    text = '\n'.join([p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 50])
                
                # Extract metadata
                meta_description = ""
                meta_desc = soup.find('meta', {'name': 'description'}) or soup.find('meta', {'property': 'og:description'})
                if meta_desc:
                    meta_description = meta_desc.get('content', '')
                
                # Extract publish date
                publish_date = None
                date_selectors = [
                    'meta[property="article:published_time"]',
                    'meta[name="publishdate"]',
                    'time[datetime]',
                    '.date',
                    '.publish-date'
                ]
                
                for selector in date_selectors:
                    date_elem = soup.select_one(selector)
                    if date_elem:
                        publish_date = date_elem.get('datetime') or date_elem.get('content') or date_elem.get_text()
                        break
                
                # Extract author
                authors = []
                author_selectors = [
                    'meta[name="author"]',
                    'meta[property="article:author"]', 
                    '.author',
                    '.byline'
                ]
                
                for selector in author_selectors:
                    author_elem = soup.select_one(selector)
                    if author_elem:
                        author = author_elem.get('content') or author_elem.get_text()
                        if author:
                            authors.append(author.strip())
                
                return {
                    "success": True,
                    "method": "beautifulsoup",
                    "title": title,
                    "text": text,
                    "authors": authors,
                    "publish_date": publish_date,
                    "url": url,
                    "summary": text[:500] + "..." if len(text) > 500 else text,
                    "word_count": len(text.split()),
                    "meta_description": meta_description,
                    "content_length": len(text),
                    "extraction_confidence": 0.8 if text else 0.3
                }
                
            except Exception as e:
                logger.warning(f"BeautifulSoup extraction failed for {url}: {e}")
                return {"error": f"Content extraction failed: {str(e)}"}
            
        except Exception as e:
            return {"error": f"URL processing failed: {str(e)}"}

# Global enhanced data source manager instance
_enhanced_data_source_manager = None

def get_enhanced_data_source_manager() -> EnhancedDataSourceManager:
    """Get or create global enhanced data source manager instance."""
    global _enhanced_data_source_manager
    if _enhanced_data_source_manager is None:
        _enhanced_data_source_manager = EnhancedDataSourceManager()
    return _enhanced_data_source_manager