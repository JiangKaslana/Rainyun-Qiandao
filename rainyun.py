import logging
import os
import random
import re
import time
import subprocess
import sys

# --- 引入通知模块 ---
try:
    # 假设 notify.py 和 rainyun.py 在同一目录下
    from notify import send
except ImportError:
    # 定义一个占位函数，防止未找到 notify.py 时脚本崩溃
    def send(title, content, **kwargs):
        print(f"通知模块未找到。标题: {title}, 内容: {content}")
        
# --- 必需的依赖 ---
import cv2
import ddddocr
import requests
from selenium import webdriver
from selenium.common import TimeoutException
from selenium.webdriver import ActionChains
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.webdriver import WebDriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
# 尝试导入 webdriver_manager
try:
    from webdriver_manager.chrome import ChromeDriverManager
    # 尝试不同的ChromeType导入路径
    try:
        from webdriver_manager.core.utils import ChromeType
    except ImportError:
        try:
            from webdriver_manager.chrome import ChromeType
        except ImportError:
            ChromeType = None
except ImportError:
    ChromeDriverManager = None
    ChromeType = None


# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 全局变量
det = None  # ddddocr的检测器

# --- 配置（从环境变量获取）---
USERS_RAW = os.environ.get("RAINYUN_USERS")
if not USERS_RAW:
    logger.error("环境变量 RAINYUN_USERS 未设置，请设置为 'user1#pass1&user2#pass2' 格式")
    # --- 失败通知 ---
    send("❌ 雨云签到失败: 配置错误", "环境变量 RAINYUN_USERS 未设置，脚本退出。")
    # --- 失败通知结束 ---
    sys.exit(1)

USERS = [user.split("#") for user in USERS_RAW.split("&")]
DEBUG = os.environ.get("RAINYUN_DEBUG", "false").lower() == "true"


def init_selenium(debug: bool = False, headless: bool = True) -> WebDriver:
    """初始化 Selenium WebDriver"""
    ops = webdriver.ChromeOptions()
    
    # 基础配置
    ops.add_argument("--no-sandbox")
    ops.add_argument("--disable-dev-shm-usage")
    ops.add_argument("--window-size=1920,1080")
    
    # 无头模式
    if headless:
        ops.add_argument("--headless")
        ops.add_argument("--disable-gpu")
        
    # 调试模式
    if debug:
        ops.add_experimental_option("detach", True)

    try:
        if ChromeDriverManager:
            # 使用 webdriver_manager 自动管理驱动
            driver_path = ChromeDriverManager().install()
            service = Service(driver_path)
            return webdriver.Chrome(service=service, options=ops)
        else:
            # 备用方式：依赖本地的 chromedriver
            if os.name == 'posix': # Linux/Mac
                service = Service("./chromedriver")
            else: # Windows
                service = Service("chromedriver.exe")
            return webdriver.Chrome(service=service, options=ops)
    except Exception as e:
        logger.error(f"初始化 WebDriver 失败: {e}")
        logger.error("请确保已安装 Chrome 浏览器，并在PATH或当前目录下有匹配版本的 chromedriver")
        raise


def download_image(driver: WebDriver, xpath: str, filename: str) -> bool:
    """从元素样式中解析 URL 并下载图片"""
    os.makedirs("temp", exist_ok=True)
    
    # 尝试直接获取 img 元素的 src
    try:
        img_element = driver.find_element(By.XPATH, xpath)
        url = img_element.get_attribute("src")
    except:
        # 尝试从 style 中获取背景图 URL
        try:
            element = driver.find_element(By.XPATH, xpath)
            style = element.get_attribute("style")
            url_match = re.search(r'url\("?(.+?)"?\)', style)
            if url_match:
                url = url_match.group(1)
            else:
                logger.error("未在元素中找到图片URL！")
                return False
        except Exception as e:
            logger.error(f"查找元素或解析样式失败: {e}")
            return False

    if not url:
        logger.error("获取到空的图片URL！")
        return False
        
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            path = os.path.join("temp", filename)
            with open(path, "wb") as f:
                f.write(response.content)
            return True
        else:
            logger.error(f"下载图片失败，状态码: {response.status_code}！")
            return False
    except Exception as e:
        logger.error(f"下载图片请求失败: {e}")
        return False


def get_sprite_pieces(sprite_path: str) -> list[str]:
    """将目标图分割成三块，返回分割后的文件名列表"""
    try:
        img = cv2.imread(sprite_path)
        h, w, _ = img.shape
        w_piece = w // 3
        
        paths = []
        for i in range(3):
            piece = img[:, i * w_piece:(i + 1) * w_piece]
            path = os.path.join("temp", f"sprite_{i + 1}.jpg")
            cv2.imwrite(path, piece)
            paths.append(path)
        return paths
    except Exception as e:
        logger.error(f"分割目标图失败: {e}")
        return []


def compute_similarity(img1_path: str, img2_path: str) -> float:
    """使用 SIFT 特征匹配计算两张图片的相似度"""
    try:
        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

        if img1 is None or img2 is None:
            return 0.0

        # 初始化 SIFT 检测器
        sift = cv2.SIFT_create()

        # 查找关键点和描述符
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
            return 0.0

        # 初始化 FLANN 匹配器
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        # 使用 KNN 匹配
        matches = flann.knnMatch(des1, des2, k=2)

        # 比例测试 (Ratio Test)
        good = []
        for pair in matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < 0.7 * n.distance:
                    good.append(m)
            elif len(pair) == 1:
                # 只有 1 个匹配项，直接接受
                good.append(pair[0])

        # 相似度基于良好匹配点的数量
        # 使用关键点少的图片的关键点数量作为归一化基准
        base = min(len(kp1), len(kp2))
        if base == 0:
            return 0.0
            
        similarity = len(good) / base
        return similarity
    except Exception as e:
        # logger.error(f"计算相似度失败: {e}") # 避免过多日志
        return 0.0


def process_captcha():
    """处理腾讯点选验证码的核心逻辑"""
    global det
    if det is None:
        det = ddddocr.DdddOcr(det=True)
        logger.info("ddddocr 目标检测器初始化完成。")

    driver = webdriver.Chrome() # 临时 WebDriver
    
    # 获取验证码元素
    try:
        slideBg = driver.find_element(By.ID, "slideBg")
        instruction = driver.find_element(By.ID, "instruction")
        confirm = driver.find_element(By.ID, "confirm")
    except Exception as e:
        logger.error(f"未找到验证码核心元素: {e}")
        return False
        
    for _ in range(5):  # 最多尝试5次
        try:
            # 1. 下载图片
            logger.info("开始下载验证码图片...")
            if not download_image(driver, '//*[@id="slideBg"]', "captcha.jpg"): continue
            if not download_image(driver, '//*[@id="instruction"]/div/img', "sprite.jpg"): continue
            
            # 2. 分割目标图标
            sprite_paths = get_sprite_pieces(os.path.join("temp", "sprite.jpg"))
            if not sprite_paths: continue
            
            # 3. 目标检测 (ddddocr)
            with open(os.path.join("temp", "captcha.jpg"), 'rb') as f:
                captcha_b = f.read()
            
            # 使用 ddddocr 进行目标检测，获取大图中所有对象的边界框
            bboxes = det.detection(captcha_b)
            
            if not bboxes:
                logger.warning("ddddocr 未检测到任何目标区域，尝试刷新验证码。")
                driver.find_element(By.ID, "reload").click()
                time.sleep(2)
                continue

            # 4. 特征匹配 (SIFT)
            logger.info(f"ddddocr 检测到 {len(bboxes)} 个目标区域。")
            
            # 存储每个目标图标的最佳匹配区域和相似度
            match_results = [] # 元素: (sprite_index, similarity, bbox)

            for i, sprite_path in enumerate(sprite_paths):
                max_similarity = 0.0
                best_bbox = None
                
                for bbox in bboxes:
                    x, y, w, h = bbox
                    # 裁剪大图中的区域
                    crop_img = cv2.imread(os.path.join("temp", "captcha.jpg"))[y:h, x:w]
                    temp_spec_path = os.path.join("temp", f"temp_spec.jpg")
                    cv2.imwrite(temp_spec_path, crop_img)
                    
                    # 计算相似度
                    similarity = compute_similarity(sprite_path, temp_spec_path)
                    
                    if similarity > max_similarity:
                        max_similarity = similarity
                        best_bbox = bbox
                
                if best_bbox:
                    match_results.append((i, max_similarity, best_bbox))
                    
            # 5. 排序和筛选最佳点击点
            match_results.sort(key=lambda x: x[1], reverse=True) # 按相似度降序排列

            # 检查是否成功找到三个最佳匹配点（确保是三个不同的点）
            final_clicks = [] # 存储最终的 (x, y) 坐标
            unique_bboxes = set()
            
            for sprite_index, similarity, bbox in match_results:
                if bbox not in unique_bboxes:
                    # 获取边界框的中心点作为点击坐标
                    center_x = (bbox[0] + bbox[2]) / 2
                    center_y = (bbox[1] + bbox[3]) / 2
                    final_clicks.append((center_x, center_y))
                    unique_bboxes.add(bbox)
                    if len(final_clicks) == 3:
                        break

            if len(final_clicks) < 3:
                logger.warning(f"只找到 {len(final_clicks)} 个不重复的点击点，尝试刷新验证码。")
                driver.find_element(By.ID, "reload").click()
                time.sleep(2)
                continue

            # 6. 坐标转换和模拟点击
            logger.info("准备点击...")
            
            # 获取背景图元素的尺寸，用于坐标映射
            bg_width_raw = slideBg.get_attribute("offsetWidth")
            bg_height_raw = slideBg.get_attribute("offsetHeight")
            
            # 假设 captcha.jpg 尺寸为 300x200（这是腾讯验证码常见尺寸）
            # 实际的图片加载后尺寸可能需要动态获取
            img_width, img_height = 300, 200 # 使用默认值

            # 尝试读取图片以获取实际尺寸
            try:
                img = cv2.imread(os.path.join("temp", "captcha.jpg"))
                img_height, img_width, _ = img.shape
            except:
                logger.warning("无法读取图片获取尺寸，使用默认 300x200 映射。")


            action = ActionChains(driver)
            for x_coord, y_coord in final_clicks:
                # 将图片像素坐标映射到浏览器元素坐标
                final_x = int(x_coord * bg_width_raw / img_width)
                final_y = int(y_coord * bg_height_raw / img_height)

                logger.info(f"点击坐标: ({final_x}, {final_y})")
                # 移动到元素，然后偏移并点击
                action.move_to_element_with_offset(slideBg, final_x, final_y).click().perform()
                time.sleep(random.uniform(0.5, 1.5)) # 模拟人类点击间隔

            # 7. 提交验证
            confirm.click()
            time.sleep(3) # 等待结果

            # 8. 检查结果
            # 检查结果元素，通常是 #tcOperation
            result_element = driver.find_element(By.ID, "tcOperation")
            if "show-success" in result_element.get_attribute("class"):
                logger.info("验证码通过！")
                return True
            else:
                logger.warning("验证码失败，尝试刷新重试。")
                driver.find_element(By.ID, "reload").click()
                time.sleep(2)
                
        except Exception as e:
            logger.error(f"验证码处理失败: {e}")
            try:
                # 尝试点击刷新以应对未知的错误
                driver.find_element(By.ID, "reload").click()
                time.sleep(2)
            except:
                pass # 刷新失败则忽略

    logger.error("多次尝试验证码均失败。")
    return False


def main():
    if not USERS:
        return

    # 仅初始化一次 WebDriver
    try:
        driver = init_selenium(debug=DEBUG, headless=True)
    except Exception:
        # init_selenium 失败时已打印日志和通知
        return

    # 引入 stealth.min.js
    try:
        with open("stealth.min.js", mode="r") as f:
            js = f.read()
        driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {"source": js})
    except FileNotFoundError:
        logger.warning("stealth.min.js 未找到，可能更容易被网站检测。")

    
    # 遍历所有用户
    for user, password in USERS:
        logger.info("-" * 50)
        logger.info(f"开始处理用户: {user}")
        
        try:
            # 1. 访问登录页面
            driver.get("https://app.rainyun.com/login")
            wait = WebDriverWait(driver, 30)

            # 2. 登录操作
            try:
                # 等待用户名输入框出现
                username_input = wait.until(EC.visibility_of_element_located((By.XPATH, '//*[@id="app"]/div[1]/div[3]/div/div/div[2]/div/div[2]/form/div[1]/div/div/input')))
                password_input = driver.find_element(By.XPATH, '//*[@id="app"]/div[1]/div[3]/div/div/div[2]/div/div[2]/form/div[2]/div/div/input')
                login_btn = driver.find_element(By.XPATH, '//*[@id="app"]/div[1]/div[3]/div/div/div[2]/div/div[2]/form/div[3]/div/button')

                username_input.send_keys(user)
                password_input.send_keys(password)
                login_btn.click()
            
            except TimeoutException:
                logger.error("登录页加载超时，请尝试延长超时时间或切换到国内网络环境！")
                # --- 添加失败通知（登录页面超时）---
                title = "❌ 雨云签到失败: 登录页面超时"
                content = f"账户: {user}\n页面加载超时，请尝试延长超时时间或切换到国内网络环境。"
                send(title, content)
                # --- 失败通知结束 ---
                continue # 跳到下一个用户

            # 3. 处理登录时的验证码（如果出现）
            try:
                # 等待验证码 iframe 出现
                wait.until(EC.frame_to_be_available_and_switch_to_it((By.ID, 'tcaptcha_iframe_dy')))
                logger.warning("触发登录验证码！")
                if not process_captcha():
                    raise Exception("登录验证码处理失败")
            except TimeoutException:
                logger.info("未触发登录验证码")
            except Exception as e:
                logger.error(f"处理登录验证码时出错: {e}")
                driver.switch_to.default_content()
                # --- 添加失败通知（登录验证码失败）---
                title = "❌ 雨云签到失败: 登录验证码失败"
                content = f"账户: {user}\n验证码处理失败，请检查程序是否有足够权限进行图像处理和模拟操作。"
                send(title, content)
                # --- 失败通知结束 ---
                continue # 跳到下一个用户

            # 切换回主文档
            driver.switch_to.default_content()
            time.sleep(5) # 等待跳转

            # 4. 验证登录状态并签到
            if "dashboard" in driver.current_url:
                logger.info("登录成功！")
                logger.info("正在转到赚取积分页")
                
                # 尝试多次访问赚取积分页面和点击签到
                success = False
                for _ in range(3):
                    try:
                        driver.get("https://app.rainyun.com/account/reward/earn")
                        driver.implicitly_wait(5)
                        
                        # 查找赚取积分按钮
                        earn = wait.until(EC.visibility_of_element_located((By.XPATH,
                                            '//*[@id="app"]/div[1]/div[3]/div[2]/div/div/div[2]/div[2]/div/div/div/div[1]/div/div[1]/div/div[1]/div/span[2]/a')))
                        
                        logger.info("点击赚取积分")
                        driver.execute_script("arguments[0].click();", earn)
                        
                        # 处理可能出现的验证码
                        try:
                            logger.info("检查是否需要签到验证码")
                            wait.until(EC.frame_to_be_available_and_switch_to_it((By.ID, "tcaptcha_iframe_dy")))
                            logger.info("处理签到验证码")
                            if not process_captcha():
                                # 签到验证码失败则不再重试签到
                                raise Exception("签到验证码处理失败")
                            driver.switch_to.default_content()
                        except TimeoutException:
                            logger.info("未触发签到验证码或验证码框架加载失败")
                            driver.switch_to.default_content()
                        
                        logger.info("赚取积分操作完成")
                        success = True
                        break # 签到流程执行完毕，退出重试循环

                    except Exception as e:
                        logger.error(f"访问赚取积分页面或点击时出错: {e}，刷新页面重试...")
                        driver.refresh()
                        time.sleep(3)
                
                # 5. 结果处理和通知
                if success:
                    driver.implicitly_wait(5)
                    # 获取积分信息
                    try:
                        points_raw = driver.find_element(By.XPATH,
                                                '//*[@id="app"]/div[1]/div[3]/div[2]/div/div/div[2]/div[1]/div[1]/div/p/div/h3').get_attribute(
                            "textContent")
                        current_points = int(''.join(re.findall(r'\d+', points_raw)))
                        
                        # --- 添加成功通知 ---
                        title = "✅ 雨云签到成功"
                        content = (
                            f"账户: {user}\n"
                            f"当前剩余积分: {current_points}\n"
                            f"价值: 约为 {current_points / 2000:.2f} 元"
                        )
                        send(title, content)
                        # --- 成功通知结束 ---

                        logger.info(f"当前剩余积分: {current_points} | 约为 {current_points / 2000:.2f} 元")
                        logger.info("任务执行成功！")
                    except Exception as e:
                        logger.error(f"签到成功后，获取积分信息失败: {e}")
                        # 仍然发送成功通知，但提示积分信息可能不准确
                        send("✅ 雨云签到完成 (积分信息获取失败)", f"账户: {user}\n已完成签到步骤，但未能读取当前积分。")

                else:
                    logger.error("多次尝试后仍无法完成签到流程。")
                    # --- 添加失败通知（签到按钮未找到或验证码失败）---
                    title = "❌ 雨云签到失败: 签到流程终止"
                    content = f"账户: {user}\n未能完成签到或验证码处理失败，请检查页面结构是否更新或网络是否正常。"
                    send(title, content)
                    # --- 失败通知结束 ---

            else:
                logger.error("登录失败！")
                # --- 添加失败通知（登录失败）---
                title = "❌ 雨云签到失败: 登录凭证错误"
                content = f"账户: {user}\n登录后未跳转到 Dashboard 页面，请检查用户名/密码是否正确。"
                send(title, content)
                # --- 失败通知结束 ---

        except Exception as overall_e:
            logger.error(f"处理用户 {user} 时发生意外错误: {overall_e}")
            # --- 添加失败通知（意外错误）---
            title = "❌ 雨云签到失败: 脚本意外错误"
            content = f"账户: {user}\n脚本运行时发生意外错误: {overall_e}"
            send(title, content)
            # --- 失败通知结束 ---
        
    driver.quit()
    logger.info("-" * 50)
    logger.info("所有用户处理完毕，脚本退出。")

if __name__ == "__main__":
    main()
